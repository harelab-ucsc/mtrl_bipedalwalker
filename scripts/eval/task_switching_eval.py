"""
scripts/eval/switching_and_individual.py
========================================

Task-SWITCHING evaluation for the V2 4-task bipedal walker. For each (model, ordered
task pair A->B) it runs many fixed-length episodes inside the PPO_BC env (RlFTEnv,
manual control) where the task switches once, mid-episode, and reports how well each
policy survives, recovers, and tracks reward across the switch:

  1. Switch success      — the episode runs N frames; at frame N/3 the task switches
                           from A (command A_c) to B (command B_c). A switch is
                           SUCCESSFUL iff the policy never terminates across all N frames.
  2. Failure timing      — for failures, whether the fall happened BEFORE the switch
                           (term_step < N/3) or AFTER it; and for after-switch failures,
                           how long the policy survived PAST the switch before falling
                           (frames + seconds; absolute frames kept too).
  3. Reward              — avg total episode reward + avg per-step reward, plus the RLFT
                           modular reward broken down per component (regularization +
                           per-task track terms), polled from ``info["reward_terms"]``.
                           The single track_gait term is split eval-side into walk / hop /
                           quiet by the active gait mode, so a walk->flamingo episode shows
                           walk-gait (segment A) and hop-gait (segment B) separately.

This script does task SWITCHING only — individual single-task behavior lives in
scripts/eval/singletask_eval.py.

Every model kind is driven through a uniform ``predict(obs_19) -> action_4`` over the
shared 19-dim RlFTEnv observation ``[14 proprio, cmd_vel, cmd_tilt, walk, flamingo, tilt]``:
  - "sb3"    : a PPO_BC checkpoint (stable_baselines3.PPO.load).
  - "torch"  : a DAgger-distilled StudentModel ({"policy": state_dict} checkpoint).
  - "hybrid" : the HybridModelV2 oracle (routes each task to its expert) — topline baseline.

Outputs (one named run directory under scripts/eval/output/):
  - meta.json                          full eval config + per-model name/description/path
  - models.md                          human-readable model description list
  - summary.csv                        wide, one row per model (overall + per-pair success)
  - switching_by_pair.csv              one row per (model, ordered pair A->B)
  - reward_components_by_pair.csv      one row per (model, pair), one col per reward term
  - chart_switch_success.png           grouped bar, success rate (%) per pair
  - chart_failure_timing.png           stacked bar, survived / fail-before / fail-after
  - chart_after_switch_survival.png    bar, mean survival past the switch (s)
  - chart_reward.png                   grouped bar, mean per-step reward per pair
  - chart_reward_components.png        modular reward breakdown per model (pooled over pairs)

Parallelism: one (model, pair, episode-chunk) per pool job, one model + env per worker,
torch pinned to 1 thread per worker. The per-experiment config (run name, episode length,
episodes per pair, switch fraction, command ranges, model list) lives in a YAML under
scripts/eval/configs/ (the SAME format as singletask_eval.py). Pick one on the command line:

    python scripts/eval/switching_and_individual.py switching_default   # the template, by name
    python scripts/eval/switching_and_individual.py path/to/custom.yaml
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
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
from mdp.bipedal_walker.tasks import GAIT, SINGLE_TASKS_GAIT, reward_mode

# =========================================
# config
#
# Structural constants and machine knobs live here; the per-experiment config
# (run name, episode length, episodes, switch fraction, command ranges, model
# list) is loaded from a YAML under scripts/eval/configs/ into an EvalConfig.

FPS = 50  # env step rate, fixed by BipedalWalker-v3

# Switch endpoints + pairs are config-driven (switch_tasks / exclude_pairs in the YAML).
# A "task unit" is a {name, bits, walk_dir} dict (same schema as the combination eval's
# task_combinations): a single task is a one-bit entry, a combination has several bits
# set. By default every ordered distinct pair of switch_tasks is run; exclude_pairs is
# an optional blacklist.


def _abbr(name: str) -> str:
    """Compact chart-axis label for a task name: initials of its _/+ separated tokens
    (walk_forward -> 'wf', walk_fwd+tilt -> 'wft'); single-token names keep 2 chars
    (tilt -> 'ti'). Cosmetic only — CSV / meta keep the full names."""
    parts = [p for p in re.split(r"[_+]", name) if p]
    return parts[0][:2] if len(parts) == 1 else "".join(p[0] for p in parts)


def pair_label(pair: tuple[str, str]) -> str:
    return f"{_abbr(pair[0])}→{_abbr(pair[1])}"


# --- machine / cosmetic knobs (not part of an experiment definition) ---
N_WORKERS = 14
PROGRESS_EVERY = 10  # episodes between progress pings

# --- output location + where named YAML configs live ---
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# Reward-component display order + per-task grouping (same convention as
# singletask_eval.py). The reward (rl_finetune_rewards.py) uses a CONSTANT term set;
# the single track_gait term is split eval-side into walk / hop / quiet by the active
# gait mode (see _gait_mode) so walk-gait and flamingo-hop read separately.
COMPONENT_GROUPS: dict[str, str] = {
    # regularization — always on, applies to every task
    "reg_hull_ang_vel": "regularization",
    "reg_joint_vel_l2": "regularization",
    "reg_vel_jerk": "regularization",
    "reg_body_height": "regularization",
    "alive": "regularization",
    "termination": "regularization",
    # task tracking — always on with task-conditioned targets
    "track_vel": "walk",  # velocity tracking
    "track_ang": "tilt",  # hull-angle tracking
    "track_gait_walk": "walk",  # leg alternation
    "track_gait_hop": "flamingo",  # single-leg hops
    "track_gait_quiet": "tilt",  # both feet planted (tilt / idle)
}
COMPONENT_ORDER = list(COMPONENT_GROUPS)


def _gait_mode(task_bits, cmd_vel: float) -> str:
    """The active gait mode (``walk`` | ``hop`` | ``quiet``) for a segment, via the
    scheme-aware ``tasks.reward_mode``. Splits the reward's single ``track_gait`` term
    into a mode-specific component. cmd_vel is needed under gait (a stationary two-leg
    segment buckets as quiet)."""
    return reward_mode(task_bits, cmd_vel, CFG.task_scheme)


@dataclass(frozen=True)
class EvalConfig:
    """One switching experiment's config, loaded from a YAML in scripts/eval/configs/.

    Shares the singletask_eval.py YAML format so the same config files (model list,
    command ranges) drive both. Frozen + primitive-only so it pickles cleanly to the
    pool workers (macOS 'spawn' ships it via the pool initializer — see _init_worker)."""

    eval_name: str
    ep_time: int  # seconds per episode -> the survival window (N frames)
    episodes_per_pair: int
    episode_chunk: int  # episodes per pool job (load balancing)
    seed_base: int  # episode i uses the same seed across models
    # obs-bit scheme ("gait" default, "onehot" legacy). Drives switch_tasks schema,
    # command sampling, gait-mode bucketing, env construction, and HybridModelV2.
    task_scheme: str
    switch_fraction: float  # fraction of the episode held by task A before the switch
    cmd_vel_forward_range: tuple[float, float]
    cmd_vel_backward_range: tuple[float, float]
    cmd_tilt_range: tuple[float, float]
    # Switch endpoints: each {name, bits (walk, flamingo, tilt), walk_dir}. Single tasks
    # are one-bit entries; combinations have several bits set. All ordered distinct pairs
    # are run by default; exclude_pairs drops specific (from, to) name pairs (blacklist).
    switch_tasks: list[dict]
    exclude_pairs: list[tuple[str, str]]
    only_models: list[str] | None  # subset of model names to run, or None for all
    date_suffix: bool  # append a timestamp to the output directory name
    flat_terrain: bool  # force perfectly flat terrain (disable height randomization); default bumpy
    # Each model dict: name, kind ("sb3"|"torch"|"hybrid"), ref (checkpoint path
    # relative to models/, None for hybrid), desc (shown in models.md / meta.json).
    models: list[dict]

    @property
    def episode_len(self) -> int:
        return self.ep_time * FPS  # frames; success = survive all of these

    @property
    def env_ep_time(self) -> int:
        return self.ep_time + 10  # env truncation kept well past the window

    @property
    def switch_step(self) -> int:
        return int(self.episode_len * self.switch_fraction)  # frame the task switches


def validate_switch_tasks(raw: list, scheme: str = GAIT) -> list[dict]:
    """Parse + validate the switch_tasks config block (scheme-aware; same schema as
    the combination eval's task_combinations). Fails fast (ValueError) on a missing/
    duplicate name or malformed task spec.

    gait: each unit is {name, label, gait [two_leg, one_leg], cmd_vel_range,
        cmd_tilt_range} → {name, label, gait_bits (two_leg, one_leg, 0),
        cmd_vel_range (tuple), cmd_tilt_range (tuple)}.
    onehot: each unit is {name, bits (walk, flamingo, tilt), walk_dir}."""
    if not raw:
        raise ValueError("switch_tasks must be a non-empty list")
    tasks: list[dict] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        name = entry.get("name")
        if not name:
            raise ValueError(f"switch_tasks[{i}]: missing 'name'")
        if name in seen:
            raise ValueError(f"switch_tasks: duplicate name {name!r}")
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
            tasks.append(
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

        tasks.append({"name": name, "bits": bits, "walk_dir": walk_dir})
    return tasks


# Default switch endpoints when a config omits switch_tasks. Onehot: the four legacy
# individual tasks (preserves the original 12-ordered-pair behavior so singletask
# configs drive the switching eval unchanged). Gait: the 5 gait single tasks.
DEFAULT_SWITCH_TASKS = [
    {"name": "walk_forward", "bits": [1, 0, 0], "walk_dir": "forward"},
    {"name": "walk_backward", "bits": [1, 0, 0], "walk_dir": "backward"},
    {"name": "flamingo", "bits": [0, 1, 0], "walk_dir": None},
    {"name": "tilt", "bits": [0, 0, 1], "walk_dir": None},
]


def _default_gait_switch_tasks() -> list[dict]:
    """The 5 gait single tasks as switch_tasks units (gait + command ranges)."""
    return [
        {
            "name": t.name,
            "label": t.name,
            "gait": list(t.gait),
            "cmd_vel_range": list(t.cmd_vel_range),
            "cmd_tilt_range": list(t.cmd_tilt_range),
        }
        for t in SINGLE_TASKS_GAIT
    ]


def build_pairs(switch_tasks: list[dict], exclude_pairs: list) -> list[tuple[str, str]]:
    """Every ordered distinct (from, to) name pair of switch_tasks, minus any listed in
    exclude_pairs (a blacklist). exclude_pairs entries must reference known task names;
    raises if the blacklist empties the pair set."""
    names = [t["name"] for t in switch_tasks]
    name_set = set(names)
    excluded: set[tuple[str, str]] = set()
    for entry in exclude_pairs or []:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            raise ValueError(f"exclude_pairs entry must be [from, to], got {entry!r}")
        a, b = entry[0], entry[1]
        for nm in (a, b):
            if nm not in name_set:
                raise ValueError(f"exclude_pairs: unknown task name {nm!r}")
        excluded.add((a, b))
    pairs = [(a, b) for a in names for b in names if a != b and (a, b) not in excluded]
    if not pairs:
        raise ValueError("no switch pairs left after applying exclude_pairs")
    return pairs


def load_config(name_or_path: str) -> EvalConfig:
    """Load an EvalConfig from YAML. ``name_or_path`` is either a path to a
    .yaml/.yml file, or a bare name resolved to scripts/eval/configs/<name>.yaml.

    ``episodes_per_pair`` defaults to the config's ``episodes_per_task``,
    ``switch_fraction`` to 1/3, and ``switch_tasks`` to the four individual tasks
    (DEFAULT_SWITCH_TASKS), so the singletask configs load unchanged."""
    p = Path(name_or_path)
    if p.suffix.lower() not in (".yaml", ".yml"):
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"eval config not found: {p}")
    with open(p) as f:
        raw = yaml.safe_load(f)
    scheme = raw.get("task_scheme", GAIT)
    default_tasks = _default_gait_switch_tasks() if scheme == GAIT else DEFAULT_SWITCH_TASKS
    return EvalConfig(
        eval_name=raw["eval_name"],
        ep_time=int(raw["ep_time"]),
        episodes_per_pair=int(
            raw["episodes_per_pair"]
            if "episodes_per_pair" in raw
            else raw["episodes_per_task"]
        ),
        episode_chunk=int(raw["episode_chunk"]),
        seed_base=int(raw["seed_base"]),
        task_scheme=scheme,
        switch_fraction=float(raw.get("switch_fraction", 1.0 / 3.0)),
        # legacy global ranges (onehot only); gait carries ranges per switch_task unit.
        cmd_vel_forward_range=tuple(raw.get("cmd_vel_forward_range", (0.0, 5.0))),
        cmd_vel_backward_range=tuple(raw.get("cmd_vel_backward_range", (-5.0, 0.0))),
        cmd_tilt_range=tuple(raw.get("cmd_tilt_range", (-0.75, 0.75))),
        switch_tasks=validate_switch_tasks(raw.get("switch_tasks") or default_tasks, scheme),
        exclude_pairs=[tuple(p) for p in (raw.get("exclude_pairs") or [])],
        only_models=raw.get("only_models"),
        date_suffix=bool(raw.get("date_suffix", True)),
        flat_terrain=bool(raw.get("flat_terrain", False)),
        models=raw["models"],
    )


# Per-worker experiment config. Set in main() and shipped to each pool worker via
# _init_worker. None until then — never read at import time.
CFG: EvalConfig = None  # type: ignore[assignment]


# =========================================
# model + path resolution


def resolve_path(spec: dict) -> str | None:
    """Resolve a model ref to a concrete checkpoint path (in the main process).

    ``ref`` is a hard-coded path relative to models/ pointing directly at the
    checkpoint to load. hybrid has no checkpoint."""
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
    rewards enabled so each task produces a meaningful modular reward."""
    return RlFTEnv(
        make("BipedalWalker-v3", render_mode=None),
        ep_time=CFG.env_ep_time,
        use_rew_for_individual_tasks=True,
        manual_ctrl_mode=True,
        task_scheme=CFG.task_scheme,
        flat_terrain=CFG.flat_terrain,
    )


def sample_combo_command(unit: dict, rng: np.random.Generator):
    """Return ``(bits, cmd_vel, cmd_tilt)`` for a switch-task unit.

    Gait: sample cmd_vel ~ U(cmd_vel_range) and cmd_tilt ~ U(cmd_tilt_range) (a
    degenerate (0,0) range pins the command to 0) and use the unit's gait_bits.
    Onehot (legacy): cmd_vel from the walk_dir range when the walk bit is set,
    cmd_tilt from the tilt range when the tilt bit is set."""
    if CFG.task_scheme == GAIT:
        bits = unit["gait_bits"]
        vel = float(rng.uniform(*unit["cmd_vel_range"]))
        tilt = float(rng.uniform(*unit["cmd_tilt_range"]))
        return bits, vel, tilt
    bits, walk_dir = unit["bits"], unit["walk_dir"]
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


def build_switch_schedule(unit_a: dict, unit_b: dict, rng: np.random.Generator):
    """[(bits, vel, tilt, n_steps)] = unit A for CFG.switch_step frames, then unit B
    for the rest of the episode. Each command is sampled ONCE per segment (held fixed,
    no within-segment modulation) so the switch is a single clean transition."""
    bits_a, vel_a, tilt_a = sample_combo_command(unit_a, rng)
    bits_b, vel_b, tilt_b = sample_combo_command(unit_b, rng)
    s = CFG.switch_step
    return [
        (bits_a, vel_a, tilt_a, s),
        (bits_b, vel_b, tilt_b, CFG.episode_len - s),
    ]


def run_episode(env: RlFTEnv, predict, schedule, seed: int) -> dict:
    """Drive the env through the two-segment switch schedule. At each segment boundary
    the task + command are snapped and the obs tail rebuilt so the first action already
    sees the new task (mirrors scripts/ppo_bc/play.py). Stops on termination (a fall).
    Returns terminated flag, the termination frame, total reward, and the per-step sum
    of each modular reward component (track_gait split by the segment's gait mode)."""
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
        gait_key = f"track_gait_{_gait_mode(bits, vel)}"  # split track_gait by this segment's mode
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
                    term_step=step,
                    total_reward=total_reward,
                    steps=step,
                    comp_sum=comp_sum,
                )
    return dict(
        terminated=False,
        term_step=None,
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
        fail_before=0,
        fail_after=0,
        after_frames_sum=0.0,  # frames survived PAST the switch (term_step - S)
        after_abs_frames_sum=0.0,  # absolute term frame (term_step)
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
    in each worker (so module-level CFG is reset to None) and does not inherit main()'s
    state — so we ship the loaded config in here and pin torch to 1 thread."""
    global CFG
    CFG = cfg
    torch.set_num_threads(1)


def _run_job(job: dict, q) -> None:
    """Pool worker: load one model + env, run a chunk of episodes for one switch pair,
    and emit progress + a partial aggregate via the queue."""
    try:
        predict = make_predict_fn(job["kind"], job["path"])
        env = make_eval_env()
        pair, n, base = job["pair"], job["n_episodes"], job["seed_base"]
        unit_a, unit_b = job["from_unit"], job["to_unit"]
        agg = _empty_agg()

        sent = 0
        for i in range(n):
            seed = base + i
            rng = np.random.default_rng(seed)
            res = run_episode(env, predict, build_switch_schedule(unit_a, unit_b, rng), seed)

            agg["n"] += 1
            agg["reward_sum"] += res["total_reward"]
            agg["steps_sum"] += res["steps"]
            for ck, cv in res["comp_sum"].items():
                agg["comp_sum"][ck] = agg["comp_sum"].get(ck, 0.0) + cv

            if not res["terminated"]:
                agg["success"] += 1
            elif res["term_step"] < CFG.switch_step:
                agg["fail_before"] += 1
            else:
                agg["fail_after"] += 1
                agg["after_frames_sum"] += res["term_step"] - CFG.switch_step
                agg["after_abs_frames_sum"] += res["term_step"]

            if (i + 1) - sent >= PROGRESS_EVERY:
                q.put(("progress", (i + 1) - sent))
                sent = i + 1
        if n - sent > 0:
            q.put(("progress", n - sent))
        env.close()
        q.put(("result", job["model"], pair, agg))
    except Exception as e:  # surface worker errors instead of hanging the pool
        q.put(("error", job["model"], job.get("pair"), repr(e)))


def _build_jobs(entries, pairs, unit_by_name):
    """Fan each (model, pair) out into CFG.episode_chunk-sized jobs. Each job carries the
    from/to task units (bits + walk_dir) so the worker can build the switch schedule
    without re-resolving names."""
    jobs = []
    for e in entries:
        for pair in pairs:
            a, b = pair
            for off in range(0, CFG.episodes_per_pair, CFG.episode_chunk):
                n = min(CFG.episode_chunk, CFG.episodes_per_pair - off)
                jobs.append(
                    dict(
                        model=e["name"],
                        kind=e["kind"],
                        path=e["path"],
                        pair=pair,
                        from_unit=unit_by_name[a],
                        to_unit=unit_by_name[b],
                        n_episodes=n,
                        seed_base=CFG.seed_base + off,
                    )
                )
    return jobs


# =========================================
# finalize + reporting


def _metrics_from_agg(a: dict) -> dict:
    """Reportable metrics for one raw aggregate (one (model, pair), or a pool of pairs).
    fail_before/after percentages are over FAILURES; frac_* are over ALL episodes
    (success + before + after = 1) for the stacked failure-timing chart."""
    n = max(a["n"], 1)
    steps = max(a["steps_sum"], 1)
    fails = a["fail_before"] + a["fail_after"]
    after_frames = a["after_frames_sum"] / a["fail_after"] if a["fail_after"] else 0.0
    after_abs = a["after_abs_frames_sum"] / a["fail_after"] if a["fail_after"] else 0.0
    comp_ps = {k: a["comp_sum"][k] / steps for k in a["comp_sum"]}
    return dict(
        n=a["n"],
        success_rate=a["success"] / n,
        fail_before=a["fail_before"],
        fail_after=a["fail_after"],
        pct_fail_before=(a["fail_before"] / fails) if fails else 0.0,
        pct_fail_after=(a["fail_after"] / fails) if fails else 0.0,
        frac_success=a["success"] / n,
        frac_fail_before=a["fail_before"] / n,
        frac_fail_after=a["fail_after"] / n,
        after_survival_frames=after_frames,
        after_survival_seconds=after_frames / FPS,
        after_abs_frames=after_abs,
        after_abs_seconds=after_abs / FPS,
        avg_total_reward=a["reward_sum"] / n,
        avg_per_step_reward=a["reward_sum"] / steps,
        comp_per_step=comp_ps,
    )


def _finalize(entries, raw, pairs):
    """Turn raw per-(model, pair) aggregates into reportable metrics. Adds a pooled
    ``overall`` per model: all pairs pooled (equal episodes per pair, so the pooled
    means equal the per-pair averages; pooling also yields a step-weighted reward)."""
    out = {}
    for e in entries:
        name = e["name"]
        per_pair = {pair: _metrics_from_agg(raw[name][pair]) for pair in pairs}
        pooled = _empty_agg()
        for pair in pairs:
            _merge_agg(pooled, raw[name][pair])
        out[name] = dict(per_pair=per_pair, overall=_metrics_from_agg(pooled))
    return out


def _print_model_list(entries):
    print("\nModels\n======")
    for i, e in enumerate(entries, 1):
        print(f"{i}. {e['name']}")
        print(f"    - {e['desc']}")


def _print_report(entries, results, pairs):
    names = [e["name"] for e in entries]
    nw = max((len(n) for n in names), default=5)

    title = f"Task switching — overall (pooled over all {len(pairs)} pairs)"
    print(f"\n{title}")
    print("=" * len(title))
    cols = ["success", "%fail_before", "%fail_after", "surv_after(s)", "reward/step"]
    print(f"{'model':<{nw}}  " + "  ".join(f"{c:>14}" for c in cols))
    for n in names:
        o = results[n]["overall"]
        cells = [
            f"{o['success_rate'] * 100:>13.1f}%",
            f"{o['pct_fail_before'] * 100:>13.1f}%",
            f"{o['pct_fail_after'] * 100:>13.1f}%",
            f"{o['after_survival_seconds']:>14.2f}",
            f"{o['avg_per_step_reward']:>14.3f}",
        ]
        print(f"{n:<{nw}}  " + "  ".join(cells))


# =========================================
# outputs: csv + charts


def _ordered_components(results):
    """Components observed (pooled overall) across all models, in COMPONENT_ORDER."""
    seen = set()
    for r in results.values():
        seen.update(r["overall"]["comp_per_step"].keys())
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


def _write_charts(entries, results, pairs, out_dir):
    names = [e["name"] for e in entries]
    pair_labels = [pair_label(p) for p in pairs] + ["overall"]
    written = []

    def per_pair_series(metric):
        return {
            n: [results[n]["per_pair"][p][metric] for p in pairs]
            + [results[n]["overall"][metric]]
            for n in names
        }

    # 1) success rate (%) per pair
    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(pair_labels)), 5))
    _grouped_bar(
        ax,
        pair_labels,
        {n: [v * 100 for v in vals] for n, vals in per_pair_series("success_rate").items()},
        "success rate (%)",
        f"Switch survival rate per pair  (episode = {CFG.ep_time}s, switch @ {CFG.switch_step}f)",
    )
    ax.set_ylim(0, 105)
    p = os.path.join(out_dir, "chart_switch_success.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)

    # 2) failure timing (overall) — stacked: survived / fail-before / fail-after
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(names)), 5))
    x = np.arange(len(names))
    succ = [results[n]["overall"]["frac_success"] * 100 for n in names]
    fb = [results[n]["overall"]["frac_fail_before"] * 100 for n in names]
    fa = [results[n]["overall"]["frac_fail_after"] * 100 for n in names]
    ax.bar(x, succ, label="survived", color="#4caf50")
    ax.bar(x, fb, bottom=succ, label="fail before switch", color="#ff9800")
    ax.bar(x, fa, bottom=[s + b for s, b in zip(succ, fb)], label="fail after switch", color="#f44336")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("% of episodes")
    ax.set_ylim(0, 105)
    ax.set_title("Failure timing (pooled over pairs)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    p = os.path.join(out_dir, "chart_failure_timing.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)

    # 3) mean survival past the switch (s), for after-switch failures (overall)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(names)), 5))
    x = np.arange(len(names))
    ax.bar(x, [results[n]["overall"]["after_survival_seconds"] for n in names], color="#5c6bc0")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("mean survival after switch (s)")
    ax.set_title("After-switch survival before termination (failures only)")
    ax.grid(axis="y", alpha=0.3)
    p = os.path.join(out_dir, "chart_after_switch_survival.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)

    # 4) mean per-step reward per pair
    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(pair_labels)), 5))
    _grouped_bar(
        ax,
        pair_labels,
        per_pair_series("avg_per_step_reward"),
        "mean reward / step",
        "Behavioral quality (RLFT reward) per switch pair",
    )
    p = os.path.join(out_dir, "chart_reward.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)

    # 5) modular reward breakdown — one chart, pooled over pairs
    comps = _ordered_components(results)
    if comps:
        fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(comps)), 5))
        comp_labels = [f"{c}\n({COMPONENT_GROUPS.get(c, '?')})" for c in comps]
        _grouped_bar(
            ax,
            comp_labels,
            {
                n: [results[n]["overall"]["comp_per_step"].get(c, 0.0) for c in comps]
                for n in names
            },
            "mean reward / step",
            "Modular reward breakdown (pooled over switch pairs)",
        )
        p = os.path.join(out_dir, "chart_reward_components.png")
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p)

    return written


def _write_outputs(entries, results, meta, pairs, out_dir):
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

    # summary.csv — wide, one row per model: pooled overall + per-pair success
    fields = [
        "model",
        "switch_succ",
        "pct_fail_before",
        "pct_fail_after",
        "after_surv_frames",
        "after_surv_sec",
        "avg_total_reward",
        "avg_per_step_reward",
    ]
    fields += [f"succ_{a}->{b}" for a, b in pairs]
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in entries:
            n = e["name"]
            o = results[n]["overall"]
            row = {
                "model": n,
                "switch_succ": round(o["success_rate"], 4),
                "pct_fail_before": round(o["pct_fail_before"], 4),
                "pct_fail_after": round(o["pct_fail_after"], 4),
                "after_surv_frames": round(o["after_survival_frames"], 2),
                "after_surv_sec": round(o["after_survival_seconds"], 3),
                "avg_total_reward": round(o["avg_total_reward"], 3),
                "avg_per_step_reward": round(o["avg_per_step_reward"], 4),
            }
            for a, b in pairs:
                row[f"succ_{a}->{b}"] = round(results[n]["per_pair"][(a, b)]["success_rate"], 4)
            w.writerow(row)
    written.append(os.path.join(out_dir, "summary.csv"))

    # switching_by_pair.csv — one row per (model, ordered pair A->B), plus pooled overall
    with open(os.path.join(out_dir, "switching_by_pair.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "pair",
                "from",
                "to",
                "n",
                "success_rate",
                "fail_before",
                "fail_after",
                "pct_fail_before",
                "pct_fail_after",
                "after_surv_frames",
                "after_surv_sec",
                "after_abs_frames",
                "after_abs_sec",
                "avg_total_reward",
                "avg_per_step_reward",
            ]
        )
        for e in entries:
            n = e["name"]
            for a, b in pairs:
                d = results[n]["per_pair"][(a, b)]
                w.writerow(
                    [
                        n,
                        f"{a}->{b}",
                        a,
                        b,
                        d["n"],
                        round(d["success_rate"], 4),
                        d["fail_before"],
                        d["fail_after"],
                        round(d["pct_fail_before"], 4),
                        round(d["pct_fail_after"], 4),
                        round(d["after_survival_frames"], 2),
                        round(d["after_survival_seconds"], 3),
                        round(d["after_abs_frames"], 2),
                        round(d["after_abs_seconds"], 3),
                        round(d["avg_total_reward"], 3),
                        round(d["avg_per_step_reward"], 4),
                    ]
                )
            # pooled overall row
            o = results[n]["overall"]
            w.writerow(
                [
                    n,
                    "overall",
                    "",
                    "",
                    o["n"],
                    round(o["success_rate"], 4),
                    o["fail_before"],
                    o["fail_after"],
                    round(o["pct_fail_before"], 4),
                    round(o["pct_fail_after"], 4),
                    round(o["after_survival_frames"], 2),
                    round(o["after_survival_seconds"], 3),
                    round(o["after_abs_frames"], 2),
                    round(o["after_abs_seconds"], 3),
                    round(o["avg_total_reward"], 3),
                    round(o["avg_per_step_reward"], 4),
                ]
            )
    written.append(os.path.join(out_dir, "switching_by_pair.csv"))

    # reward_components_by_pair.csv — one row per (model, pair), one col per component
    all_comps = [c for c in COMPONENT_ORDER]
    extra = set()
    for r in results.values():
        for p in pairs:
            extra.update(r["per_pair"][p]["comp_per_step"].keys())
    all_comps += sorted(c for c in extra if c not in COMPONENT_ORDER)
    with open(
        os.path.join(out_dir, "reward_components_by_pair.csv"), "w", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["model", "pair"] + all_comps)
        for e in entries:
            n = e["name"]
            for a, b in pairs:
                comp = results[n]["per_pair"][(a, b)]["comp_per_step"]
                w.writerow(
                    [n, f"{a}->{b}"]
                    + [round(comp[c], 5) if c in comp else "" for c in all_comps]
                )
    written.append(os.path.join(out_dir, "reward_components_by_pair.csv"))

    written += _write_charts(entries, results, pairs, out_dir)

    print("\nWrote:")
    for p in written:
        print(f"  {p}")


# =========================================


def main(cfg: EvalConfig, config_source: str):
    global CFG
    CFG = cfg  # main-process config; workers get their own copy via _init_worker

    # config-driven switch endpoints + the ordered pairs to run (all pairs minus the
    # exclude_pairs blacklist); unit_by_name carries each task's bits + walk_dir.
    unit_by_name = {t["name"]: t for t in cfg.switch_tasks}
    pairs = build_pairs(cfg.switch_tasks, cfg.exclude_pairs)

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

    jobs = _build_jobs(entries, pairs, unit_by_name)
    total_episodes = sum(j["n_episodes"] for j in jobs)
    n_jobs = len(jobs)

    raw = {e["name"]: {p: _empty_agg() for p in pairs} for e in entries}

    print(f"Config:      {config_source}")
    print(f"Eval name:   {cfg.eval_name}")
    print(f"Models:      {len(entries)}  ({', '.join(e['name'] for e in entries)})")
    print(
        f"Switch tasks:{len(cfg.switch_tasks)}  "
        f"({', '.join(t['name'] for t in cfg.switch_tasks)})"
    )
    print(f"Pairs:       {len(pairs)} ordered task pairs")
    print(
        f"Episode:     {cfg.episode_len} frames ({cfg.ep_time}s), "
        f"switch at frame {cfg.switch_step} ({cfg.switch_fraction:.3f})"
    )
    print(f"Episodes:    {cfg.episodes_per_pair:,} per (model, pair)")
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
                        _, model, pair, err = msg
                        errors.append((model, pair, err))
                        completed += 1
                        pbar.set_postfix(done=f"{completed}/{n_jobs}", errors=len(errors))
                    else:
                        _, model, pair, agg = msg
                        _merge_agg(raw[model][pair], agg)
                        completed += 1
                        pbar.set_postfix(done=f"{completed}/{n_jobs}", errors=len(errors))
            pool.join()

    if errors:
        print(f"\n!! {len(errors)} job(s) errored:")
        for model, pair, err in errors[:20]:
            print(f"   {model} {pair}: {err}")

    results = _finalize(entries, raw, pairs)
    _print_report(entries, results, pairs)

    stamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    base_name = f"{cfg.eval_name}"
    dir_name = f"{base_name}_{stamp}" if cfg.date_suffix else base_name
    out_dir = os.path.join(OUTPUT_ROOT, dir_name)
    meta = dict(
        eval_name=cfg.eval_name,
        config_source=config_source,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        episode_len_frames=cfg.episode_len,
        episode_len_sec=cfg.ep_time,
        switch_fraction=cfg.switch_fraction,
        switch_step=cfg.switch_step,
        episodes_per_pair=cfg.episodes_per_pair,
        fps=FPS,
        switch_tasks=cfg.switch_tasks,
        exclude_pairs=[list(p) for p in cfg.exclude_pairs],
        pairs=[f"{a}->{b}" for a, b in pairs],
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
        errors=[f"{m} {p}: {err}" for m, p, err in errors],
    )
    _write_outputs(entries, results, meta, pairs, out_dir)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="__array_wrap__ must accept context and return_scalar arguments",
        category=DeprecationWarning,
    )
    available = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
    parser = argparse.ArgumentParser(
        description="Task-switching eval. Pick an experiment config under "
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
