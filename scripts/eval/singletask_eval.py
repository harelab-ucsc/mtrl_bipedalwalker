"""
scripts/eval/singletask_eval.py
===============================

Per-task, per-model behavioral evaluation for the V2 4-task bipedal walker. For
each (model, task) it runs many fixed-length episodes inside the PPO_BC env
(RlFTEnv, manual control) and reports three families of metrics:

  1. Time alive          — mean +/- std survival per episode (frames and seconds).
  2. Success rate        — fraction of episodes that survive the full episode window.
  3. Behavioral quality  — the RLFT modular reward, broken down per component
                           (stability + per-task tracking/gait terms), polled live
                           from ``info["reward_terms"]`` and averaged per step.

Every model kind is driven through a uniform ``predict(obs_19) -> action_4`` over the
shared 19-dim RlFTEnv observation ``[14 proprio, cmd_vel, cmd_tilt, walk, flamingo, tilt]``:

  - "sb3"    : a PPO_BC checkpoint, loaded with stable_baselines3.PPO.load (inference
               needs no experts/task_bits — see scripts/ppo_bc/play.py).
  - "torch"  : a DAgger-distilled StudentModel (loads a {"policy": state_dict} checkpoint).
  - "hybrid" : the HybridModelV2 oracle (routes each task to its dedicated expert) — a
               topline reference baseline.

Outputs (one timestamped, named run directory under scripts/eval/output/):
  - meta.json                        full eval config + per-model name/description/path
  - models.md                        human-readable model description list
  - summary.csv                      wide, one row per model (at-a-glance comparison)
  - by_task.csv                      one row per (model, task): alive/success/reward
  - reward_components_by_task.csv    one row per (model, task), one col per reward term
  - chart_time_alive.png             grouped bar, mean seconds alive (+/- std)
  - chart_success_rate.png           grouped bar, success rate (%)
  - chart_reward.png                 grouped bar, mean per-step reward
  - chart_reward_components_<task>.png   per task: modular reward breakdown per model

by_task.csv also gets a synthetic ``overall`` task per model: the four real tasks
pooled (equal episodes per task -> the macro/micro averages coincide), giving a
single across-task row alongside the per-task ones.

Parallelism: one (model, task, episode-chunk) per pool job, one model + env per worker,
torch pinned to 1 thread per worker. Defaults to os.cpu_count() workers to saturate the CPU.

The per-experiment config (run name, episode length, episodes per task, command
ranges, model list, ...) lives in a YAML under scripts/eval/configs/. Structural
constants (FPS, the four tasks, reward component grouping) and machine knobs
(N_WORKERS, PROGRESS_EVERY) stay in this file. Pick a config on the command line:

    python scripts/eval/singletask_eval.py            # configs/default.yaml
    python scripts/eval/singletask_eval.py smoke      # configs/smoke.yaml (by name)
    python scripts/eval/singletask_eval.py path/to/custom.yaml
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render charts to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from gymnasium import make
from tqdm import tqdm

from utils.paths import MODELS_DIR
from wrappers.ppo_bc.ppo_bc_env import RlFTEnv
from mdp.bipedal_walker.student import StudentModel
from mdp.bipedal_walker.hybrid import HybridModelV2

# =========================================
# config
#
# Structural constants and machine knobs live here; the per-experiment config
# (run name, episode length, episodes, command ranges, model list) is loaded
# from a YAML under scripts/eval/configs/ into an EvalConfig — see load_config.

FPS = 50  # env step rate, fixed by BipedalWalker-v3

# --- the four individual tasks (codebase convention) ---
TASKS = ["walk_forward", "walk_backward", "flamingo", "tilt"]

# --- machine / cosmetic knobs (not part of an experiment definition) ---
N_WORKERS = 14 * 2
PROGRESS_EVERY = 10  # episodes between progress pings

# --- output location + where named YAML configs live ---
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class EvalConfig:
    """One experiment's eval config, loaded from a YAML in scripts/eval/configs/.

    Holds only the per-experiment knobs; structural constants (FPS, TASKS,
    COMPONENT_GROUPS) and machine knobs (N_WORKERS, PROGRESS_EVERY) stay as
    module-level constants. Frozen + primitive-only so it pickles cleanly to the
    pool workers (macOS uses the 'spawn' start method, so each worker re-imports
    this module and receives the config via the pool initializer rather than
    inheriting it — see _init_worker)."""

    eval_name: str
    ep_time: int  # seconds per episode -> the survival window
    modulate_period: int  # re-sample the task command every N seconds
    episodes_per_task: int
    episode_chunk: int  # episodes per pool job (load balancing)
    seed_base: int  # episode i uses the same seed across models
    cmd_vel_forward_range: tuple[float, float]
    cmd_vel_backward_range: tuple[float, float]
    cmd_tilt_range: tuple[float, float]
    only_models: list[str] | None  # subset of model names to run, or None for all
    # Each model dict: name, kind ("sb3"|"torch"|"hybrid"), ref (checkpoint path
    # relative to models/, None for hybrid — the *latest* checkpoint, never
    # best/best_model.zip), desc (shown in models.md / meta.json).
    models: list[dict]

    # --- derived (frames; depend on FPS) ---
    @property
    def episode_len(self) -> int:
        return self.ep_time * FPS  # frames; success = survive all of these

    @property
    def env_ep_time(self) -> int:
        return self.ep_time + 10  # env truncation kept well past the window

    @property
    def modulate_every(self) -> int:
        return self.modulate_period * FPS  # re-sampling period in frames


def load_config(name_or_path: str) -> EvalConfig:
    """Load an EvalConfig from YAML. ``name_or_path`` is either a path to a
    .yaml/.yml file, or a bare name resolved to scripts/eval/configs/<name>.yaml."""
    p = Path(name_or_path)
    if p.suffix.lower() not in (".yaml", ".yml"):
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"eval config not found: {p}")
    with open(p) as f:
        raw = yaml.safe_load(f)
    return EvalConfig(
        eval_name=raw["eval_name"],
        ep_time=int(raw["ep_time"]),
        modulate_period=int(raw["modulate_period"]),
        episodes_per_task=int(raw["episodes_per_task"]),
        episode_chunk=int(raw["episode_chunk"]),
        seed_base=int(raw["seed_base"]),
        cmd_vel_forward_range=tuple(raw["cmd_vel_forward_range"]),
        cmd_vel_backward_range=tuple(raw["cmd_vel_backward_range"]),
        cmd_tilt_range=tuple(raw["cmd_tilt_range"]),
        only_models=raw.get("only_models"),
        models=raw["models"],
    )


# Per-worker experiment config. Set in the main process (main) and shipped to
# each pool worker via _init_worker. None until then — never read at import time
# (typed as EvalConfig since every read happens after it's set).
CFG: EvalConfig = None  # type: ignore[assignment]

# Stable display order for the modular reward components (union across tasks), each
# tagged with the task group(s) it scores so it's clear which task a term belongs to.
# Stability terms are always on (every task), so they list all three; the rest are
# task-specific. Components absent for a given task simply don't appear for that task's
# rows/charts. ``COMPONENT_GROUPS`` is the single source of truth for both order and
# grouping; the group label is surfaced in the per-task charts and meta.json.
COMPONENT_GROUPS: dict[str, str] = {
    # stability — always on, applies to every task
    "hull_ang_vel": "stability",
    "joint_vel_l2": "stability",
    "body_height": "stability",
    "termination": "stability",
    # velocity tracking (walk)
    "vel_tracking": "walk",
    "vel_tracking_fine": "walk",
    "vel_jerk": "walk",
    # walk gait
    "leg_alt_bonus": "walk",
    "hopping_penalty": "walk",
    # tilt tracking
    "hull_ang_tracking": "tilt",
    "hull_ang_tracking_fine": "tilt",
    # flamingo gait
    "leg_alt_penalty": "flamingo",
    "hopping_bonus": "flamingo",
    "both_leg_contact": "flamingo",
}
COMPONENT_ORDER = list(COMPONENT_GROUPS)

# =========================================
# model + path resolution


def resolve_path(spec: dict) -> str | None:
    """Resolve a model ref to a concrete checkpoint path (in the main process).

    ``ref`` is a hard-coded path relative to models/ pointing directly at the
    checkpoint to load (the latest one we want — never best/best_model.zip).
    hybrid has no checkpoint."""
    if spec["kind"] == "hybrid":
        return None
    p = MODELS_DIR / spec["ref"]
    if not p.exists():
        raise FileNotFoundError(f"{spec['name']}: {p} not found")
    return str(p)


def make_predict_fn(kind: str, path: str | None):
    """Return a uniform ``predict(obs_19) -> action_4`` for the given model kind.
    All three kinds consume the same 19-dim RlFTEnv observation."""
    if kind == "sb3":
        from stable_baselines3 import PPO

        model = PPO.load(path, device="cpu")  # type: ignore

        def predict(obs):
            return model.predict(obs, deterministic=True)[0]

        return predict

    if kind == "torch":
        model = StudentModel()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # type: ignore
        model.load_state_dict(ckpt["policy"])
        model.eval()

        def predict(obs):  # type: ignore
            with torch.no_grad():
                return model(torch.tensor(obs, dtype=torch.float32)).numpy()

        return predict

    if kind == "hybrid":
        model = HybridModelV2()

        def predict(obs):
            with torch.no_grad():
                return model.forward(torch.tensor(obs, dtype=torch.float32)).numpy()

        return predict

    raise ValueError(f"unknown model kind: {kind}")


# =========================================
# env + rollout


def make_eval_env() -> RlFTEnv:
    """Single RlFTEnv driven manually (no internal resampling), with task-specific
    rewards enabled so individual tasks produce a meaningful modular reward."""
    return RlFTEnv(
        make("BipedalWalker-v3", render_mode=None),
        ep_time=CFG.env_ep_time,
        use_rew_for_individual_tasks=True,
        manual_ctrl_mode=True,
    )


def sample_command(task: str, rng: np.random.Generator):
    """Return ``(task_bits, cmd_vel, cmd_tilt)`` for a task. Walk direction is the
    velocity sign (>= 0 -> forward). Flamingo has no command; tilt commands an angle."""
    if task == "walk_forward":
        return (1, 0, 0), float(rng.uniform(*CFG.cmd_vel_forward_range)), 0.0
    if task == "walk_backward":
        return (1, 0, 0), float(rng.uniform(*CFG.cmd_vel_backward_range)), 0.0
    if task == "flamingo":
        return (0, 1, 0), 0.0, 0.0
    if task == "tilt":
        return (0, 0, 1), 0.0, float(rng.uniform(*CFG.cmd_tilt_range))
    raise ValueError(task)


def build_schedule(task: str, rng: np.random.Generator):
    """Hold one task for CFG.episode_len frames, re-sampling its command every
    CFG.modulate_every frames (a no-op for flamingo, which has no command)."""
    segs = []
    remaining = CFG.episode_len
    while remaining > 0:
        n = min(CFG.modulate_every, remaining)
        bits, vel, tilt = sample_command(task, rng)
        segs.append((bits, vel, tilt, n))
        remaining -= n
    return segs


def run_episode(env: RlFTEnv, predict, schedule, seed: int) -> dict:
    """Drive the env through a schedule of (task_bits, cmd_vel, cmd_tilt, n_steps)
    segments. At each segment boundary the task + command are snapped and the obs
    tail rebuilt so the first action already sees the new command (mirrors
    scripts/ppo_bc/play.py). Stops on termination (a fall). Returns survival frames,
    success, total reward, and the per-step sum of each modular reward component."""
    np.random.seed(
        seed & 0x7FFFFFFF
    )  # RlFTEnv.reset uses global np.random for hull init
    obs, _ = env.reset(seed=seed)

    step = 0
    total_reward = 0.0
    comp_sum: dict[str, float] = {}
    for bits, vel, tilt, n_steps in schedule:
        env._task_id_vec = bits
        env._cmd_vec = (vel, tilt)
        env._cmd_vec_target = (vel, tilt)
        # rebuild the trailing cmd+task obs slots for the new segment
        obs = env._derive_full_obs(obs[:-5], env._effective_cmd_vec(), bits)
        for _ in range(n_steps):
            action = predict(obs)
            obs, rew, term, _trunc, info = env.step(action)
            total_reward += float(rew)
            for k, v in info["reward_terms"].items():
                comp_sum[k] = comp_sum.get(k, 0.0) + float(v)
            step += 1
            if term:
                return dict(
                    terminated=True,
                    frames_alive=step,
                    total_reward=total_reward,
                    steps=step,
                    comp_sum=comp_sum,
                )
    return dict(
        terminated=False,
        frames_alive=CFG.episode_len,
        total_reward=total_reward,
        steps=step,
        comp_sum=comp_sum,
    )


# =========================================
# aggregation (mergeable across chunks, no per-episode storage)


def _empty_agg() -> dict:
    return dict(
        n=0,
        success=0,
        alive_sum=0.0,
        alive_sq_sum=0.0,
        reward_sum=0.0,
        steps_sum=0,
        comp_sum={},
    )


def _merge_agg(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if k == "comp_sum":
            d = dst.setdefault("comp_sum", {})
            for ck, cv in v.items():
                d[ck] = d.get(ck, 0.0) + cv
        else:
            dst[k] = dst.get(k, 0) + v


def _init_worker(cfg: EvalConfig) -> None:
    """Pool initializer: runs once per worker. macOS 'spawn' re-imports this module
    in each worker (so module-level CFG is reset to None) and does not inherit
    main()'s state — so we ship the loaded config in here and pin torch to 1 thread."""
    global CFG
    CFG = cfg
    torch.set_num_threads(1)


def _run_job(job: dict, q) -> None:
    """Pool worker: load one model + env, run a chunk of episodes for one task,
    and emit progress + a partial aggregate via the queue."""
    try:
        predict = make_predict_fn(job["kind"], job["path"])
        env = make_eval_env()
        task, n, base = job["task"], job["n_episodes"], job["seed_base"]
        agg = _empty_agg()

        sent = 0
        for i in range(n):
            seed = base + i
            rng = np.random.default_rng(seed)
            res = run_episode(env, predict, build_schedule(task, rng), seed)

            agg["n"] += 1
            agg["success"] += int(not res["terminated"])
            agg["alive_sum"] += res["frames_alive"]
            agg["alive_sq_sum"] += res["frames_alive"] ** 2
            agg["reward_sum"] += res["total_reward"]
            agg["steps_sum"] += res["steps"]
            for ck, cv in res["comp_sum"].items():
                agg["comp_sum"][ck] = agg["comp_sum"].get(ck, 0.0) + cv

            if (i + 1) - sent >= PROGRESS_EVERY:
                q.put(("progress", (i + 1) - sent))
                sent = i + 1
        if n - sent > 0:
            q.put(("progress", n - sent))
        env.close()
        q.put(("result", job["model"], task, agg))
    except Exception as e:  # surface worker errors instead of hanging the pool
        q.put(("error", job["model"], job.get("task"), repr(e)))


def _build_jobs(entries):
    """Fan each (model, task) out into CFG.episode_chunk-sized jobs."""
    jobs = []
    for e in entries:
        for task in TASKS:
            for off in range(0, CFG.episodes_per_task, CFG.episode_chunk):
                n = min(CFG.episode_chunk, CFG.episodes_per_task - off)
                jobs.append(
                    dict(
                        model=e["name"],
                        kind=e["kind"],
                        path=e["path"],
                        task=task,
                        n_episodes=n,
                        seed_base=CFG.seed_base + off,
                    )
                )
    return jobs


# =========================================
# finalize + reporting


def _metrics_from_agg(a: dict) -> dict:
    """Reportable metrics for one raw aggregate (one (model, task), or a pool of
    tasks). Works for the synthetic ``overall`` too: pooling the raw aggregates
    gives a true across-task mean and a proper std (not a std-of-means)."""
    n = max(a["n"], 1)
    steps = max(a["steps_sum"], 1)
    mean = a["alive_sum"] / n
    var = max(a["alive_sq_sum"] / n - mean**2, 0.0)
    std = float(np.sqrt(var))
    comp_ps = {k: a["comp_sum"][k] / steps for k in a["comp_sum"]}
    return dict(
        n=a["n"],
        success_rate=a["success"] / n,
        alive_mean_frames=mean,
        alive_std_frames=std,
        alive_mean_sec=mean / FPS,
        alive_std_sec=std / FPS,
        avg_total_reward=a["reward_sum"] / n,
        avg_per_step_reward=a["reward_sum"] / steps,
        comp_per_step=comp_ps,
    )


def _finalize(entries, raw):
    """Turn raw per-(model, task) aggregates into reportable metrics. Adds a
    synthetic ``overall`` task per model: the four real tasks pooled. With equal
    episodes per task the pooled means equal the per-task averages; pooling also
    yields a meaningful overall std and (for per-step reward) a step-weighted mean."""
    out = {}
    for e in entries:
        name = e["name"]
        per_task = {task: _metrics_from_agg(raw[name][task]) for task in TASKS}

        pooled = _empty_agg()
        for task in TASKS:
            _merge_agg(pooled, raw[name][task])
        per_task["overall"] = _metrics_from_agg(pooled)

        out[name] = dict(
            per_task=per_task,
            overall_success=per_task["overall"]["success_rate"],
            overall_alive_sec=per_task["overall"]["alive_mean_sec"],
            overall_reward_ps=per_task["overall"]["avg_per_step_reward"],
        )
    return out


def _print_model_list(entries):
    """Print the evaluated models as a Notion-pasteable numbered list:
    ``N. name`` followed by an indented ``- desc`` bullet."""
    print("\nModels\n======")
    for i, e in enumerate(entries, 1):
        print(f"{i}. {e['name']}")
        print(f"    - {e['desc']}")


def _print_report(entries, results):
    names = [e["name"] for e in entries]
    nw = max((len(n) for n in names), default=5)

    def hdr(title):
        print(f"\n{title}\n{'=' * len(title)}")

    hdr("Time alive — mean seconds (std)")
    print(f"{'model':<{nw}}  " + "  ".join(f"{t:>16}" for t in TASKS + ["overall"]))
    for n in names:
        r = results[n]
        cells = [
            f"{r['per_task'][t]['alive_mean_sec']:>6.2f} ({r['per_task'][t]['alive_std_sec']:>4.2f})"
            for t in TASKS
        ]
        cells.append(f"{r['overall_alive_sec']:>16.2f}")
        print(f"{n:<{nw}}  " + "  ".join(f"{c:>16}" for c in cells))

    hdr("Success rate (%)")
    print(f"{'model':<{nw}}  " + "  ".join(f"{t:>14}" for t in TASKS + ["overall"]))
    for n in names:
        r = results[n]
        cells = [f"{r['per_task'][t]['success_rate'] * 100:>13.1f}%" for t in TASKS]
        cells.append(f"{r['overall_success'] * 100:>13.1f}%")
        print(f"{n:<{nw}}  " + "  ".join(cells))

    hdr("Behavioral quality — mean per-step reward")
    print(f"{'model':<{nw}}  " + "  ".join(f"{t:>14}" for t in TASKS + ["overall"]))
    for n in names:
        r = results[n]
        cells = [f"{r['per_task'][t]['avg_per_step_reward']:>14.3f}" for t in TASKS]
        cells.append(f"{r['overall_reward_ps']:>14.3f}")
        print(f"{n:<{nw}}  " + "  ".join(cells))


# =========================================
# outputs: csv + charts


def _ordered_components(results, task):
    """Components observed for a task across all models, in COMPONENT_ORDER."""
    seen = set()
    for r in results.values():
        seen.update(r["per_task"][task]["comp_per_step"].keys())
    ordered = [c for c in COMPONENT_ORDER if c in seen]
    ordered += sorted(
        c for c in seen if c not in COMPONENT_ORDER
    )  # any unexpected term
    return ordered


def _grouped_bar(ax, group_labels, series, ylabel, title, errors=None):
    """series: {model_name -> [value per group]}. errors: optional {model_name -> [std]}."""
    n_groups = len(group_labels)
    n_series = max(len(series), 1)
    x = np.arange(n_groups)
    width = 0.8 / n_series
    for i, (name, vals) in enumerate(series.items()):
        offs = x + (i - (n_series - 1) / 2) * width
        err = errors[name] if errors else None
        ax.bar(offs, vals, width, label=name, yerr=err, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=min(n_series, 3))


def _write_charts(entries, results, out_dir):
    names = [e["name"] for e in entries]
    written = []

    # 1) time alive (seconds, with std error bars)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(TASKS)), 5))
    _grouped_bar(
        ax,
        TASKS,
        {
            n: [results[n]["per_task"][t]["alive_mean_sec"] for t in TASKS]
            for n in names
        },
        "mean time alive (s)",
        f"Time alive per task  (episode = {CFG.ep_time}s)",
        errors={
            n: [results[n]["per_task"][t]["alive_std_sec"] for t in TASKS]
            for n in names
        },
    )
    p = os.path.join(out_dir, "chart_time_alive.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)

    # 2) success rate (%)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(TASKS)), 5))
    _grouped_bar(
        ax,
        TASKS,
        {
            n: [results[n]["per_task"][t]["success_rate"] * 100 for t in TASKS]
            for n in names
        },
        "success rate (%)",
        f"Full-episode survival rate per task  (episode = {CFG.ep_time}s)",
    )
    ax.set_ylim(0, 105)
    p = os.path.join(out_dir, "chart_success_rate.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)

    # 3) behavioral quality: mean per-step reward
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(TASKS)), 5))
    _grouped_bar(
        ax,
        TASKS,
        {
            n: [results[n]["per_task"][t]["avg_per_step_reward"] for t in TASKS]
            for n in names
        },
        "mean reward / step",
        "Behavioral quality (RLFT reward) per task",
    )
    p = os.path.join(out_dir, "chart_reward.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)

    # 4) modular reward breakdown — one chart per task
    for task in TASKS:
        comps = _ordered_components(results, task)
        if not comps:
            continue
        fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(comps)), 5))
        # annotate each component with its task group so the chart makes clear
        # which task a reward term belongs to
        comp_labels = [f"{c}\n({COMPONENT_GROUPS.get(c, '?')})" for c in comps]
        _grouped_bar(
            ax,
            comp_labels,
            {
                n: [
                    results[n]["per_task"][task]["comp_per_step"].get(c, 0.0)
                    for c in comps
                ]
                for n in names
            },
            "mean reward / step",
            f"Modular reward breakdown — {task}",
        )
        p = os.path.join(out_dir, f"chart_reward_components_{task}.png")
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p)

    return written


def _write_outputs(entries, results, meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    written = []

    # meta.json — full config + per-model name/description/path
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    written.append(os.path.join(out_dir, "meta.json"))

    # models.md — human-readable model description list
    with open(os.path.join(out_dir, "models.md"), "w") as f:
        f.write(f"# Models evaluated — {meta['eval_name']} ({meta['timestamp']})\n\n")
        f.write("| name | kind | description | checkpoint |\n")
        f.write("| --- | --- | --- | --- |\n")
        for m in meta["models"]:
            f.write(
                f"| `{m['name']}` | {m['kind']} | {m['desc']} | `{m['path'] or '-'}` |\n"
            )
    written.append(os.path.join(out_dir, "models.md"))

    # summary.csv — wide, one row per model
    fields = ["model"]
    for t in TASKS:
        fields += [f"alive_sec_{t}", f"succ_{t}", f"reward_ps_{t}"]
    fields += ["alive_sec_overall", "succ_overall", "reward_ps_overall"]
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in entries:
            n = e["name"]
            r = results[n]
            row = {"model": n}
            for t in TASKS:
                pt = r["per_task"][t]
                row[f"alive_sec_{t}"] = round(pt["alive_mean_sec"], 3)
                row[f"succ_{t}"] = round(pt["success_rate"], 4)
                row[f"reward_ps_{t}"] = round(pt["avg_per_step_reward"], 4)
            row["alive_sec_overall"] = round(r["overall_alive_sec"], 3)
            row["succ_overall"] = round(r["overall_success"], 4)
            row["reward_ps_overall"] = round(r["overall_reward_ps"], 4)
            w.writerow(row)
    written.append(os.path.join(out_dir, "summary.csv"))

    # by_task.csv — one row per (model, task), plus a pooled "overall" row per model
    with open(os.path.join(out_dir, "by_task.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "task",
                "n",
                "success_rate",
                "alive_mean_frames",
                "alive_std_frames",
                "alive_mean_sec",
                "alive_std_sec",
                "avg_total_reward",
                "avg_per_step_reward",
            ]
        )
        for e in entries:
            n = e["name"]
            for t in TASKS + ["overall"]:
                pt = results[n]["per_task"][t]
                w.writerow(
                    [
                        n,
                        t,
                        pt["n"],
                        round(pt["success_rate"], 4),
                        round(pt["alive_mean_frames"], 2),
                        round(pt["alive_std_frames"], 2),
                        round(pt["alive_mean_sec"], 3),
                        round(pt["alive_std_sec"], 3),
                        round(pt["avg_total_reward"], 3),
                        round(pt["avg_per_step_reward"], 4),
                    ]
                )
    written.append(os.path.join(out_dir, "by_task.csv"))

    # reward_components_by_task.csv — one row per (model, task), one col per component
    all_comps = [c for c in COMPONENT_ORDER]
    extra = set()
    for r in results.values():
        for t in TASKS:
            extra.update(r["per_task"][t]["comp_per_step"].keys())
    all_comps += sorted(c for c in extra if c not in COMPONENT_ORDER)
    with open(
        os.path.join(out_dir, "reward_components_by_task.csv"), "w", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["model", "task"] + all_comps)
        for e in entries:
            n = e["name"]
            for t in TASKS:
                comp = results[n]["per_task"][t]["comp_per_step"]
                w.writerow(
                    [n, t] + [round(comp[c], 5) if c in comp else "" for c in all_comps]
                )
    written.append(os.path.join(out_dir, "reward_components_by_task.csv"))

    written += _write_charts(entries, results, out_dir)

    print("\nWrote:")
    for p in written:
        print(f"  {p}")


# =========================================


def main(cfg: EvalConfig, config_source: str):
    global CFG
    CFG = cfg  # main-process config; workers get their own copy via _init_worker

    # optional model filter (cfg.only_models) for quick subsets / smoke runs
    specs = cfg.models
    if cfg.only_models:
        wanted = set(cfg.only_models)
        specs = [m for m in cfg.models if m["name"] in wanted]

    # resolve checkpoint paths up front (fail fast / skip missing models)
    entries = []
    for spec in specs:
        try:
            path = resolve_path(spec)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue
        entries.append(
            dict(name=spec["name"], kind=spec["kind"], path=path, desc=spec["desc"])
        )

    if not entries:
        print("No models resolved — nothing to evaluate.")
        return

    jobs = _build_jobs(entries)
    total_episodes = sum(j["n_episodes"] for j in jobs)
    n_jobs = len(jobs)

    raw = {e["name"]: {t: _empty_agg() for t in TASKS} for e in entries}

    print(f"Config:      {config_source}")
    print(f"Eval name:   {cfg.eval_name}")
    print(f"Models:      {len(entries)}  ({', '.join(e['name'] for e in entries)})")
    print(f"Tasks:       {len(TASKS)}  ({', '.join(TASKS)})")
    print(
        f"Episode:     {cfg.episode_len} frames ({cfg.ep_time}s), command modulated every {cfg.modulate_every}"
    )
    print(f"Episodes:    {cfg.episodes_per_task:,} per (model, task)")
    print(
        f"Jobs:        {n_jobs}   workers: {N_WORKERS}   total episodes: {total_episodes:,}"
    )

    _print_model_list(entries)

    completed = 0
    errors = []
    with mp.Manager() as manager:
        q = manager.Queue()
        with mp.Pool(
            processes=N_WORKERS, initializer=_init_worker, initargs=(cfg,)
        ) as pool:
            for job in jobs:
                pool.apply_async(_run_job, args=(job, q))
            pool.close()

            with tqdm(total=total_episodes, desc="Evaluating", unit="ep") as pbar:
                while completed < n_jobs:
                    msg = q.get()
                    if msg[0] == "progress":
                        pbar.update(msg[1])
                    elif msg[0] == "error":
                        _, model, task, err = msg
                        errors.append((model, task, err))
                        completed += 1
                        pbar.set_postfix(
                            done=f"{completed}/{n_jobs}", errors=len(errors)
                        )
                    else:
                        _, model, task, agg = msg
                        _merge_agg(raw[model][task], agg)
                        completed += 1
                        pbar.set_postfix(
                            done=f"{completed}/{n_jobs}", errors=len(errors)
                        )
            pool.join()

    if errors:
        print(f"\n!! {len(errors)} job(s) errored:")
        for model, task, err in errors[:20]:
            print(f"   {model} {task}: {err}")

    results = _finalize(entries, raw)
    _print_report(entries, results)

    stamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    out_dir = os.path.join(OUTPUT_ROOT, f"{cfg.eval_name}_{stamp}")
    meta = dict(
        eval_name=cfg.eval_name,
        config_source=config_source,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        episode_len_frames=cfg.episode_len,
        episode_len_sec=cfg.ep_time,
        modulate_every=cfg.modulate_every,
        episodes_per_task=cfg.episodes_per_task,
        fps=FPS,
        tasks=TASKS,
        cmd_vel_forward_range=cfg.cmd_vel_forward_range,
        cmd_vel_backward_range=cfg.cmd_vel_backward_range,
        cmd_tilt_range=cfg.cmd_tilt_range,
        component_groups=COMPONENT_GROUPS,
        n_workers=N_WORKERS,
        seed_base=cfg.seed_base,
        models=[
            dict(name=e["name"], kind=e["kind"], path=e["path"], desc=e["desc"])
            for e in entries
        ],
        errors=[f"{m} {t}: {err}" for m, t, err in errors],
    )
    _write_outputs(entries, results, meta, out_dir)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="__array_wrap__ must accept context and return_scalar arguments",
        category=DeprecationWarning,
    )
    parser = argparse.ArgumentParser(
        description="Per-task behavioral eval. Pick an experiment config under "
        "scripts/eval/configs/ (by name or path)."
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="config name resolved to "
        "scripts/eval/configs/<name>.yaml, or a path to a .yaml file",
    )
    a = parser.parse_args()
    main(load_config(a.config), config_source=a.config)
