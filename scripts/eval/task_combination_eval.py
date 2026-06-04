"""
scripts/eval/task_combination_eval.py
=====================================

Per-combination, per-model behavioral evaluation for the V2 bipedal walker. A
task combination is a 3-bit one-hot (walk, flamingo, tilt) with one or more bits
set, e.g. walk+tilt (1,0,1) or flamingo+tilt (0,1,1); single-bit entries are
allowed as baselines. For each (model, combination) it runs many fixed-length
episodes in the PPO_BC env (RlFTEnv, manual control) and reports the same metric
families as single_task_eval.py, now per combination:

  1. Time alive          — mean +/- std survival per episode (frames and seconds).
  2. Success rate        — fraction of episodes surviving the full episode window.
  3. Reward              — mean total and mean per-step RLFT reward.
  4. Reward components   — the modular reward broken down per term, polled from
                           info["reward_terms"] and averaged per step (track_gait
                           split eval-side into walk/hop/quiet by the active mode).

The combinations to test are listed explicitly in the YAML config (no auto
combinatorics). Pick a config on the command line:

    python scripts/eval/task_combination_eval.py combination_default
    python scripts/eval/task_combination_eval.py path/to/custom.yaml
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
from mdp.bipedal_walker.tasks import GAIT, ONEHOT, GaitTask, reward_mode

# =========================================
# config

FPS = 50  # env step rate, fixed by BipedalWalker-v3

N_WORKERS = int(14 * 1.5)
PROGRESS_EVERY = 10  # episodes between progress pings

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class EvalConfig:
    """One experiment's eval config, loaded from a YAML in scripts/eval/configs/.

    task_combinations is the per-combination axis: a list of {name, bits, walk_dir}
    dicts (parsed/validated by validate_combinations). Frozen + primitive-only so it
    pickles cleanly to the spawn-started pool workers (see _init_worker)."""

    eval_name: str
    ep_time: int  # seconds per episode -> the survival window
    modulate_period: int  # re-sample the command every N seconds
    episodes_per_combo: int
    episode_chunk: int  # episodes per pool job (load balancing)
    seed_base: int  # episode i uses the same seed across models
    # obs-bit scheme ("gait" default, "onehot" legacy). Drives command sampling,
    # gait-mode bucketing, env construction, and HybridModelV2 routing.
    task_scheme: str
    cmd_vel_forward_range: tuple[float, float]
    cmd_vel_backward_range: tuple[float, float]
    cmd_tilt_range: tuple[float, float]
    # Per-combination units (parsed/validated by validate_combinations).
    #   onehot: {"name", "bits": (walk, flamingo, tilt), "walk_dir": fwd|bwd|None}
    #   gait:   {"name", "label", "gait_bits": (two_leg, one_leg, 0),
    #            "cmd_vel_range", "cmd_tilt_range"}
    task_combinations: list[dict]
    only_models: list[str] | None
    date_suffix: bool
    models: list[dict]  # name, kind ("sb3"|"torch"|"hybrid"), ref, desc

    @property
    def episode_len(self) -> int:
        return self.ep_time * FPS

    @property
    def env_ep_time(self) -> int:
        return self.ep_time + 10

    @property
    def modulate_every(self) -> int:
        return self.modulate_period * FPS

    @property
    def label_by_combo(self) -> dict[str, str]:
        """Display label per combo name (gait units carry one; onehot displays
        under its own name)."""
        return {c["name"]: c.get("label", c["name"]) for c in self.task_combinations}


def validate_combinations(raw: list, scheme: str = GAIT) -> list[dict]:
    """Parse + validate the task_combinations config block (scheme-aware). Fails
    fast (ValueError) on a missing/duplicate name or malformed task spec.

    onehot: each unit is {name, bits (walk, flamingo, tilt), walk_dir} — bits must
        be length-3 0/1 with at least one set; a walk unit needs a direction.
    gait: each unit is {name, label, gait [two_leg, one_leg], cmd_vel_range,
        cmd_tilt_range} → {name, label, gait_bits (two_leg, one_leg, 0),
        cmd_vel_range (tuple), cmd_tilt_range (tuple)}."""
    if not raw:
        raise ValueError("task_combinations must be a non-empty list")
    combos: list[dict] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        name = entry.get("name")
        if not name:
            raise ValueError(f"task_combinations[{i}]: missing 'name'")
        if name in seen:
            raise ValueError(f"task_combinations: duplicate name {name!r}")
        seen.add(name)

        if scheme == GAIT:
            gait = entry.get("gait")
            if not (
                isinstance(gait, (list, tuple))
                and len(gait) == 2
                and all(g in (0, 1) for g in gait)
            ):
                raise ValueError(
                    f"{name}: gait must be a length-2 list of 0/1, got {gait!r}"
                )
            combos.append(
                {
                    "name": name,
                    "label": entry.get("label", name),
                    "gait_bits": (int(gait[0]), int(gait[1]), 0),
                    "cmd_vel_range": tuple(entry["cmd_vel_range"]),
                    "cmd_tilt_range": tuple(entry["cmd_tilt_range"]),
                }
            )
            continue

        bits = entry.get("bits")
        if not (
            isinstance(bits, (list, tuple))
            and len(bits) == 3
            and all(b in (0, 1) for b in bits)
        ):
            raise ValueError(f"{name}: bits must be a length-3 list of 0/1, got {bits!r}")
        bits = tuple(int(b) for b in bits)
        if sum(bits) == 0:
            raise ValueError(f"{name}: at least one task bit must be set")

        walk_dir = entry.get("walk_dir")
        if bits[0]:
            if walk_dir not in ("forward", "backward"):
                raise ValueError(
                    f"{name}: walk bit is set, so walk_dir must be 'forward' or "
                    f"'backward' (got {walk_dir!r})"
                )
        else:
            walk_dir = None  # ignored when the walk bit is unset

        combos.append({"name": name, "bits": bits, "walk_dir": walk_dir})
    return combos


def load_config(name_or_path: str) -> EvalConfig:
    """Load an EvalConfig from YAML. ``name_or_path`` is either a path to a
    .yaml/.yml file, or a bare name resolved to scripts/eval/configs/<name>.yaml.

    ``task_scheme`` defaults to "gait"; under gait each task_combinations unit uses
    the gait + command-range schema, under onehot the legacy bits + walk_dir schema."""
    p = Path(name_or_path)
    if p.suffix.lower() not in (".yaml", ".yml"):
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"eval config not found: {p}")
    with open(p) as f:
        raw = yaml.safe_load(f)
    scheme = raw.get("task_scheme", GAIT)
    return EvalConfig(
        eval_name=raw["eval_name"],
        ep_time=int(raw["ep_time"]),
        modulate_period=int(raw["modulate_period"]),
        episodes_per_combo=int(raw["episodes_per_combo"]),
        episode_chunk=int(raw["episode_chunk"]),
        seed_base=int(raw["seed_base"]),
        task_scheme=scheme,
        cmd_vel_forward_range=tuple(raw.get("cmd_vel_forward_range", (0.0, 5.0))),
        cmd_vel_backward_range=tuple(raw.get("cmd_vel_backward_range", (-5.0, 0.0))),
        cmd_tilt_range=tuple(raw.get("cmd_tilt_range", (-0.75, 0.75))),
        task_combinations=validate_combinations(raw["task_combinations"], scheme),
        only_models=raw.get("only_models"),
        date_suffix=bool(raw.get("date_suffix", True)),
        models=raw["models"],
    )


# Set in main(), shipped to each pool worker via _init_worker. None until then.
CFG: EvalConfig = None  # type: ignore[assignment]

# Stable display order for the modular reward components, each tagged with the task
# it scores. The reward emits a constant term set (always-on regularization + the
# three task-tracking channels); the single track_gait term is split eval-side into
# walk/hop/quiet by the active gait mode (see _gait_mode). For combinations several
# tracking terms are simultaneously meaningful (e.g. walk+tilt -> track_vel + track_ang).
COMPONENT_GROUPS: dict[str, str] = {
    "reg_hull_ang_vel": "regularization",
    "reg_joint_vel_l2": "regularization",
    "reg_vel_jerk": "regularization",
    "reg_body_height": "regularization",
    "alive": "regularization",
    "termination": "regularization",
    "track_vel": "walk",
    "track_ang": "tilt",
    "track_gait_walk": "walk",
    "track_gait_hop": "flamingo",
    "track_gait_quiet": "tilt",
}
COMPONENT_ORDER = list(COMPONENT_GROUPS)


def _gait_mode(task_bits, cmd_vel: float) -> str:
    """Active gait mode (``walk`` | ``hop`` | ``quiet``) for a segment, via the
    scheme-aware ``tasks.reward_mode``. cmd_vel is needed under gait (a stationary
    two-leg segment buckets as quiet)."""
    return reward_mode(task_bits, cmd_vel, CFG.task_scheme)


def _safe(name: str) -> str:
    """Filesystem-safe combo name for chart filenames (the human name keeps '+')."""
    return name.replace("+", "__")


# =========================================
# model + path resolution


def resolve_path(spec: dict) -> str | None:
    """Resolve a model ref to a concrete checkpoint path (hybrid has none)."""
    if spec["kind"] == "hybrid":
        return None
    p = MODELS_DIR / spec["ref"]
    if not p.exists():
        raise FileNotFoundError(f"{spec['name']}: {p} not found")
    return str(p)


def make_predict_fn(kind: str, path: str | None):
    """Return a uniform ``predict(obs_19) -> action_4`` for the given model kind."""
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
        model = HybridModelV2(scheme=CFG.task_scheme)

        def predict(obs):
            with torch.no_grad():
                return model.forward(torch.tensor(obs, dtype=torch.float32)).numpy()

        return predict

    raise ValueError(f"unknown model kind: {kind}")


# =========================================
# env + rollout


def make_eval_env() -> RlFTEnv:
    """Single RlFTEnv driven manually (no internal resampling), with task-specific
    rewards enabled so every combination produces a meaningful modular reward."""
    return RlFTEnv(
        make("BipedalWalker-v3", render_mode=None),
        ep_time=CFG.env_ep_time,
        use_rew_for_individual_tasks=True,
        manual_ctrl_mode=True,
        task_scheme=CFG.task_scheme,
    )


def sample_combo_command(combo: dict, rng: np.random.Generator):
    """Return ``(bits, cmd_vel, cmd_tilt)`` for a combination unit.

    Gait: sample cmd_vel ~ U(cmd_vel_range) and cmd_tilt ~ U(cmd_tilt_range) (a
    degenerate (0,0) range pins the command to 0) and use the unit's gait_bits.
    Onehot (legacy): cmd_vel from the walk_dir range when the walk bit is set,
    cmd_tilt from the tilt range when the tilt bit is set."""
    if CFG.task_scheme == GAIT:
        bits = combo["gait_bits"]
        vel = float(rng.uniform(*combo["cmd_vel_range"]))
        tilt = float(rng.uniform(*combo["cmd_tilt_range"]))
        return bits, vel, tilt
    bits, walk_dir = combo["bits"], combo["walk_dir"]
    vel = 0.0
    tilt = 0.0
    if bits[0]:
        rng_range = (
            CFG.cmd_vel_forward_range
            if walk_dir == "forward"
            else CFG.cmd_vel_backward_range
        )
        vel = float(rng.uniform(*rng_range))
    if bits[2]:
        tilt = float(rng.uniform(*CFG.cmd_tilt_range))
    return bits, vel, tilt


def build_combo_schedule(combo: dict, rng: np.random.Generator):
    """Hold one combination for CFG.episode_len frames, re-sampling its command
    every CFG.modulate_every frames. Bits are constant; only the command moves."""
    segs = []
    remaining = CFG.episode_len
    while remaining > 0:
        n = min(CFG.modulate_every, remaining)
        bits, vel, tilt = sample_combo_command(combo, rng)
        segs.append((bits, vel, tilt, n))
        remaining -= n
    return segs


def run_episode(env: RlFTEnv, predict, schedule, seed: int) -> dict:
    """Drive the env through a schedule of (task_bits, cmd_vel, cmd_tilt, n_steps)
    segments. At each boundary the task + command are snapped and the obs tail
    rebuilt so the first action already sees the new command. Stops on termination."""
    np.random.seed(seed & 0x7FFFFFFF)  # RlFTEnv.reset uses global np.random for hull init
    obs, _ = env.reset(seed=seed)

    step = 0
    total_reward = 0.0
    comp_sum: dict[str, float] = {}
    for bits, vel, tilt, n_steps in schedule:
        env._task_id_vec = bits
        env._cmd_vec = (vel, tilt)
        env._cmd_vec_target = (vel, tilt)
        obs = env._derive_full_obs(obs[:-5], env._effective_cmd_vec(), bits)
        gait_key = f"track_gait_{_gait_mode(bits, vel)}"
        for _ in range(n_steps):
            action = predict(obs)
            obs, rew, term, _trunc, info = env.step(action)
            total_reward += float(rew)
            for k, v in info["reward_terms"].items():
                if k == "track_gait":
                    k = gait_key
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
    global CFG
    CFG = cfg
    torch.set_num_threads(1)


def _run_job(job: dict, q) -> None:
    """Pool worker: load one model + env, run a chunk of episodes for one
    combination, and emit progress + a partial aggregate via the queue."""
    try:
        predict = make_predict_fn(job["kind"], job["path"])
        env = make_eval_env()
        combo_spec = job["combo_spec"]
        combo = job["combo"]
        n, base = job["n_episodes"], job["seed_base"]
        agg = _empty_agg()

        sent = 0
        for i in range(n):
            seed = base + i
            rng = np.random.default_rng(seed)
            res = run_episode(
                env, predict, build_combo_schedule(combo_spec, rng), seed
            )

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
        q.put(("result", job["model"], combo, agg))
    except Exception as e:  # surface worker errors instead of hanging the pool
        q.put(("error", job["model"], job.get("combo"), repr(e)))


def _build_jobs(entries, combo_specs):
    """Fan each (model, combination) out into CFG.episode_chunk-sized jobs."""
    jobs = []
    for e in entries:
        for c in combo_specs:
            for off in range(0, CFG.episodes_per_combo, CFG.episode_chunk):
                n = min(CFG.episode_chunk, CFG.episodes_per_combo - off)
                jobs.append(
                    dict(
                        model=e["name"],
                        kind=e["kind"],
                        path=e["path"],
                        combo=c["name"],
                        combo_spec=c,
                        n_episodes=n,
                        seed_base=CFG.seed_base + off,
                    )
                )
    return jobs


# =========================================
# finalize + reporting


def _metrics_from_agg(a: dict) -> dict:
    """Reportable metrics for one raw aggregate (one (model, combo), or a pool)."""
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


def _finalize(entries, raw, combos):
    """Turn raw per-(model, combo) aggregates into reportable metrics, adding a
    synthetic ``overall`` per model: all combos pooled (equal episodes -> the pooled
    means equal the per-combo averages, with a meaningful overall std)."""
    out = {}
    for e in entries:
        name = e["name"]
        per_combo = {c: _metrics_from_agg(raw[name][c]) for c in combos}

        pooled = _empty_agg()
        for c in combos:
            _merge_agg(pooled, raw[name][c])
        per_combo["overall"] = _metrics_from_agg(pooled)

        out[name] = dict(
            per_combo=per_combo,
            overall_success=per_combo["overall"]["success_rate"],
            overall_alive_sec=per_combo["overall"]["alive_mean_sec"],
            overall_reward_ps=per_combo["overall"]["avg_per_step_reward"],
        )
    return out


def _print_model_list(entries):
    print("\nModels\n======")
    for i, e in enumerate(entries, 1):
        print(f"{i}. {e['name']}")
        print(f"    - {e['desc']}")


def _print_report(entries, results, combos):
    names = [e["name"] for e in entries]
    nw = max((len(n) for n in names), default=5)
    cols = combos + ["overall"]
    cw = max(14, *(len(c) for c in cols))

    def hdr(title):
        print(f"\n{title}\n{'=' * len(title)}")

    hdr("Time alive — mean seconds (std)")
    print(f"{'model':<{nw}}  " + "  ".join(f"{c:>{cw}}" for c in cols))
    for n in names:
        r = results[n]
        cells = [
            f"{r['per_combo'][c]['alive_mean_sec']:.2f} ({r['per_combo'][c]['alive_std_sec']:.2f})"
            for c in combos
        ]
        cells.append(f"{r['overall_alive_sec']:.2f}")
        print(f"{n:<{nw}}  " + "  ".join(f"{x:>{cw}}" for x in cells))

    hdr("Success rate (%)")
    print(f"{'model':<{nw}}  " + "  ".join(f"{c:>{cw}}" for c in cols))
    for n in names:
        r = results[n]
        cells = [f"{r['per_combo'][c]['success_rate'] * 100:.1f}%" for c in combos]
        cells.append(f"{r['overall_success'] * 100:.1f}%")
        print(f"{n:<{nw}}  " + "  ".join(f"{x:>{cw}}" for x in cells))

    hdr("Behavioral quality — mean per-step reward")
    print(f"{'model':<{nw}}  " + "  ".join(f"{c:>{cw}}" for c in cols))
    for n in names:
        r = results[n]
        cells = [f"{r['per_combo'][c]['avg_per_step_reward']:.3f}" for c in combos]
        cells.append(f"{r['overall_reward_ps']:.3f}")
        print(f"{n:<{nw}}  " + "  ".join(f"{x:>{cw}}" for x in cells))


# =========================================
# outputs: csv + charts


def _ordered_components(results, combo):
    """Components observed for a combo across all models, in COMPONENT_ORDER."""
    seen = set()
    for r in results.values():
        seen.update(r["per_combo"][combo]["comp_per_step"].keys())
    ordered = [c for c in COMPONENT_ORDER if c in seen]
    ordered += sorted(c for c in seen if c not in COMPONENT_ORDER)
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


def _write_charts(entries, results, combos, out_dir):
    names = [e["name"] for e in entries]
    written = []

    # 1) time alive (seconds, with std error bars)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(combos)), 5))
    _grouped_bar(
        ax,
        combos,
        {
            n: [results[n]["per_combo"][c]["alive_mean_sec"] for c in combos]
            for n in names
        },
        "mean time alive (s)",
        f"Time alive per combination  (episode = {CFG.ep_time}s)",
        errors={
            n: [results[n]["per_combo"][c]["alive_std_sec"] for c in combos]
            for n in names
        },
    )
    p = os.path.join(out_dir, "chart_time_alive.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)

    # 2) success rate (%)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(combos)), 5))
    _grouped_bar(
        ax,
        combos,
        {
            n: [results[n]["per_combo"][c]["success_rate"] * 100 for c in combos]
            for n in names
        },
        "success rate (%)",
        f"Full-episode survival rate per combination  (episode = {CFG.ep_time}s)",
    )
    ax.set_ylim(0, 105)
    p = os.path.join(out_dir, "chart_success_rate.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)

    # 3) behavioral quality: mean per-step reward
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(combos)), 5))
    _grouped_bar(
        ax,
        combos,
        {
            n: [results[n]["per_combo"][c]["avg_per_step_reward"] for c in combos]
            for n in names
        },
        "mean reward / step",
        "Behavioral quality (RLFT reward) per combination",
    )
    p = os.path.join(out_dir, "chart_reward.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)

    # 4) modular reward breakdown — one chart per combination
    for combo in combos:
        comps = _ordered_components(results, combo)
        if not comps:
            continue
        fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(comps)), 5))
        comp_labels = [f"{c}\n({COMPONENT_GROUPS.get(c, '?')})" for c in comps]
        _grouped_bar(
            ax,
            comp_labels,
            {
                n: [
                    results[n]["per_combo"][combo]["comp_per_step"].get(c, 0.0)
                    for c in comps
                ]
                for n in names
            },
            "mean reward / step",
            f"Modular reward breakdown — {combo}",
        )
        p = os.path.join(out_dir, f"chart_reward_components_{_safe(combo)}.png")
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p)

    return written


def _write_outputs(entries, results, meta, combos, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    written = []

    # meta.json — full config + per-model name/description/path
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    written.append(os.path.join(out_dir, "meta.json"))

    # models.md — human-readable model description list
    with open(os.path.join(out_dir, "models.md"), "w") as f:
        f.write(f"# Models evaluated — {meta['eval_name']} ({meta['timestamp']})\n\n")
        f.write(
            "Note: the `hybrid` baseline has no combination expert — it averages the "
            "actions of each active task's single-task expert.\n\n"
        )
        f.write("| name | kind | description | checkpoint |\n")
        f.write("| --- | --- | --- | --- |\n")
        for m in meta["models"]:
            f.write(
                f"| `{m['name']}` | {m['kind']} | {m['desc']} | `{m['path'] or '-'}` |\n"
            )
    written.append(os.path.join(out_dir, "models.md"))

    # summary.csv — wide, one row per model
    fields = ["model"]
    for c in combos:
        fields += [f"alive_sec_{c}", f"succ_{c}", f"reward_ps_{c}"]
    fields += ["alive_sec_overall", "succ_overall", "reward_ps_overall"]
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in entries:
            n = e["name"]
            r = results[n]
            row = {"model": n}
            for c in combos:
                pc = r["per_combo"][c]
                row[f"alive_sec_{c}"] = round(pc["alive_mean_sec"], 3)
                row[f"succ_{c}"] = round(pc["success_rate"], 4)
                row[f"reward_ps_{c}"] = round(pc["avg_per_step_reward"], 4)
            row["alive_sec_overall"] = round(r["overall_alive_sec"], 3)
            row["succ_overall"] = round(r["overall_success"], 4)
            row["reward_ps_overall"] = round(r["overall_reward_ps"], 4)
            w.writerow(row)
    written.append(os.path.join(out_dir, "summary.csv"))

    # by_combo.csv — one row per (model, combo), plus a pooled "overall" row per model
    with open(os.path.join(out_dir, "by_combo.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "combination",
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
            for c in combos + ["overall"]:
                pc = results[n]["per_combo"][c]
                w.writerow(
                    [
                        n,
                        c,
                        pc["n"],
                        round(pc["success_rate"], 4),
                        round(pc["alive_mean_frames"], 2),
                        round(pc["alive_std_frames"], 2),
                        round(pc["alive_mean_sec"], 3),
                        round(pc["alive_std_sec"], 3),
                        round(pc["avg_total_reward"], 3),
                        round(pc["avg_per_step_reward"], 4),
                    ]
                )
    written.append(os.path.join(out_dir, "by_combo.csv"))

    # reward_components_by_combo.csv — one row per (model, combo), one col per component
    all_comps = [c for c in COMPONENT_ORDER]
    extra = set()
    for r in results.values():
        for c in combos:
            extra.update(r["per_combo"][c]["comp_per_step"].keys())
    all_comps += sorted(c for c in extra if c not in COMPONENT_ORDER)
    with open(
        os.path.join(out_dir, "reward_components_by_combo.csv"), "w", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["model", "combination"] + all_comps)
        for e in entries:
            n = e["name"]
            for c in combos:
                comp = results[n]["per_combo"][c]["comp_per_step"]
                w.writerow(
                    [n, c] + [round(comp[k], 5) if k in comp else "" for k in all_comps]
                )
    written.append(os.path.join(out_dir, "reward_components_by_combo.csv"))

    written += _write_charts(entries, results, combos, out_dir)

    print("\nWrote:")
    for p in written:
        print(f"  {p}")


# =========================================


def main(cfg: EvalConfig, config_source: str):
    global CFG
    CFG = cfg  # main-process config; workers get their own copy via _init_worker

    combo_specs = cfg.task_combinations
    combos = [c["name"] for c in combo_specs]

    specs = cfg.models
    if cfg.only_models:
        wanted = set(cfg.only_models)
        specs = [m for m in cfg.models if m["name"] in wanted]

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

    jobs = _build_jobs(entries, combo_specs)
    total_episodes = sum(j["n_episodes"] for j in jobs)
    n_jobs = len(jobs)

    raw = {e["name"]: {c: _empty_agg() for c in combos} for e in entries}

    print(f"Config:        {config_source}")
    print(f"Eval name:     {cfg.eval_name}")
    print(f"Models:        {len(entries)}  ({', '.join(e['name'] for e in entries)})")
    print(f"Combinations:  {len(combos)}  ({', '.join(combos)})")
    print(
        f"Episode:       {cfg.episode_len} frames ({cfg.ep_time}s), command modulated every {cfg.modulate_every}"
    )
    print(f"Episodes:      {cfg.episodes_per_combo:,} per (model, combination)")
    print(
        f"Jobs:          {n_jobs}   workers: {N_WORKERS}   total episodes: {total_episodes:,}"
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
                        _, model, combo, err = msg
                        errors.append((model, combo, err))
                        completed += 1
                        pbar.set_postfix(
                            done=f"{completed}/{n_jobs}", errors=len(errors)
                        )
                    else:
                        _, model, combo, agg = msg
                        _merge_agg(raw[model][combo], agg)
                        completed += 1
                        pbar.set_postfix(
                            done=f"{completed}/{n_jobs}", errors=len(errors)
                        )
            pool.join()

    if errors:
        print(f"\n!! {len(errors)} job(s) errored:")
        for model, combo, err in errors[:20]:
            print(f"   {model} {combo}: {err}")

    results = _finalize(entries, raw, combos)
    _print_report(entries, results, combos)

    stamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    dir_name = f"{cfg.eval_name}_{stamp}" if cfg.date_suffix else cfg.eval_name
    out_dir = os.path.join(OUTPUT_ROOT, dir_name)
    meta = dict(
        eval_name=cfg.eval_name,
        config_source=config_source,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        episode_len_frames=cfg.episode_len,
        episode_len_sec=cfg.ep_time,
        modulate_every=cfg.modulate_every,
        episodes_per_combo=cfg.episodes_per_combo,
        fps=FPS,
        task_combinations=combo_specs,
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
        errors=[f"{m} {c}: {err}" for m, c, err in errors],
    )
    _write_outputs(entries, results, meta, combos, out_dir)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="__array_wrap__ must accept context and return_scalar arguments",
        category=DeprecationWarning,
    )
    available = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
    parser = argparse.ArgumentParser(
        description="Per-combination behavioral eval. Pick an experiment config under "
        "scripts/eval/configs/ (by name or path).\n\n"
        f"Available configs: {', '.join(available)}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="config name resolved to scripts/eval/configs/<name>.yaml, "
        "or a path to a .yaml file. "
        f"Available: {', '.join(available)}",
    )
    a = parser.parse_args()
    if a.config is None:
        parser.print_help()
        print(f"\nAvailable configs: {', '.join(available)}")
        raise SystemExit(1)
    main(load_config(a.config), config_source=a.config)
