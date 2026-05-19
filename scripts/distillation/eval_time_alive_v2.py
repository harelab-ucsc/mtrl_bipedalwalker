import multiprocessing as mp
import warnings

import numpy as np
import torch
from gymnasium import make
from tqdm import tqdm

from utils.paths import MODELS_DIR
from wrappers.bipedal_walker.distill_env import DistillEnv
from mdp.bipedal_walker.student import (
    StudentModelV2,
    StudentModelXSV2,
    StudentModelSV2,
    StudentModelMV2,
    StudentModelMLV2,
    StudentModelLV2,
    StudentModelXLV2,
    StudentModelXLLV2,
    StudentModelXLLLV2,
)
from mdp.bipedal_walker.hybrid import HybridModelV2

# Tasks match the distillation V2 setup:
#   0: walk_forward   cmd_vel > 0,  cmd_tilt = 0
#   1: walk_backward  cmd_vel < 0,  cmd_tilt = 0
#   2: flamingo       cmd_vel = 0,  cmd_tilt = 0  (hop_backward expert @ 0)
#   3: tilt           cmd_vel = 0,  cmd_tilt varies
TASK_NAMES = ["walk_forward", "walk_backward", "flamingo", "tilt"]
T_EVAL = 1_000_000
UPDATE_INTERVAL = 500

MODEL_CLASSES = {
    "xs":     StudentModelXSV2,
    "s":      StudentModelSV2,
    "m":      StudentModelMV2,
    "ml":     StudentModelMLV2,
    "l":      StudentModelLV2,
    "xl":     StudentModelXLV2,
    "xll":    StudentModelXLLV2,
    "xlll":   StudentModelXLLLV2,
    "hybrid": HybridModelV2,
}

# (display_name, subdir under MODELS_DIR, model_key)
# model_key must match a key in MODEL_CLASSES above.
# For hybrid, subdir is unused — pass "".
MODELS: list[tuple[str, str, str]] = [
    ("ml.1.n02.mix",    "distill_v2/ml.1.n02.mix",    "ml"),
    ("ml.1.n02.nomix",  "distill_v2/ml.1.n02.nomix",  "ml"),
    ("ml.1.1.n02.mix",  "distill_v2/ml.1.1.n02.mix",  "ml"),
    ("ml.1.1.n02.nomix","distill_v2/ml.1.1.n02.nomix","ml"),
    ("hybrid",          "",                            "hybrid"),
]

TASKS = list(range(4))


def _configure_env(env: DistillEnv, task_id: int, mix: bool):
    if task_id == 0:  # walk forward
        env.config_hull_reset(x_range=(0.0, 40.0))
        env.config_cmd_vel(sample_range=(0.0, 5.0), interp_time=0.5, switch_time=3, zero_prob=0.2)
        if mix:
            env.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
        else:
            env.config_cmd_tilt(zero_prob=1)
    elif task_id == 1:  # walk backward
        env.config_hull_reset(x_range=(40.0, 80.0))
        env.config_cmd_vel(sample_range=(-5.0, 0.0), interp_time=0.5, switch_time=3, zero_prob=0.2)
        if mix:
            env.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
        else:
            env.config_cmd_tilt(zero_prob=1)
    elif task_id == 2:  # flamingo
        env.config_hull_reset(x_range=(20.0, 60.0))
        if mix:
            env.config_cmd_vel(sample_range=(-5.0, 5.0), switch_time=3, interp_time=0.5, zero_prob=0.2)
            env.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
        else:
            env.config_cmd_vel(zero_prob=1)
            env.config_cmd_tilt(zero_prob=1)
    elif task_id == 3:  # tilt
        env.config_hull_reset(x_range=(20.0, 60.0))
        env.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
        if mix:
            env.config_cmd_vel(sample_range=(-5.0, 5.0), switch_time=3, interp_time=0.5, zero_prob=0.2)
        else:
            env.config_cmd_vel(zero_prob=1)


def _eval_task(display_name: str, model_key: str, model_path: str, task_id: int, mix: bool, q) -> None:
    cls = MODEL_CLASSES[model_key]
    model = cls()
    if model_key != "hybrid":
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["policy"])
        model.eval()

    env = DistillEnv(make("BipedalWalker-v3", render_mode=None), ep_time=10)
    _configure_env(env, task_id, mix)

    obs, info = env.reset()
    cmd_vel = info["cmd"]["x_vel"]
    cmd_tilt = info["cmd"]["tilt"]
    done = False
    time_alive: list[float] = []
    alive = 0
    last_sent = 0

    with torch.no_grad():
        for step in range(T_EVAL):
            if done:
                time_alive.append(alive)
                obs, info = env.reset()
                cmd_vel = info["cmd"]["x_vel"]
                cmd_tilt = info["cmd"]["tilt"]
                done = False
                alive = 0
            else:
                alive += 1

            obs_s = StudentModelV2.obs(obs, task_id, cmd_vel, cmd_tilt)
            pred = model.forward(torch.tensor(obs_s, dtype=torch.float32))
            obs, _, term, trunc, info = env.step(pred.numpy())
            cmd_vel = info["cmd"]["x_vel"]
            cmd_tilt = info["cmd"]["tilt"]
            done = term or trunc

            if (step + 1) % UPDATE_INTERVAL == 0:
                q.put(UPDATE_INTERVAL)
                last_sent = step + 1

    remaining = T_EVAL - last_sent
    if remaining > 0:
        q.put(remaining)

    time_alive.append(alive)
    env.close()
    q.put(("result", display_name, task_id, mix, float(np.mean(time_alive))))


def _print_table(title: str, entries: list[tuple[str, str, str]], results: dict):
    rows = []
    for display_name, _, _ in entries:
        task_map = results[display_name]
        task_vals = [task_map.get(i) for i in range(len(TASK_NAMES))]
        evaluated = [v for v in task_vals if v is not None]
        avg = float(np.mean(evaluated)) if evaluated else 0.0
        rows.append((display_name, avg, task_vals))
    rows.sort(key=lambda r: r[1], reverse=True)

    name_w = max((len(name) for name, *_ in rows), default=5)
    val_w = 10
    task_col_w = max(len(t) for t in TASK_NAMES)

    header_parts = [f"{'Model':<{name_w}}", f"{'Avg Alive':>{val_w}}"] + [
        f"{t:>{task_col_w}}" for t in TASK_NAMES
    ]
    header = "  ".join(header_parts)
    sep = "=" * len(header)

    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for display_name, avg, task_vals in rows:
        cells = [
            f"{'-':>{task_col_w}}" if v is None else f"{v:>{task_col_w}.1f}"
            for v in task_vals
        ]
        row_parts = [f"{display_name:<{name_w}}", f"{avg:>{val_w}.1f}"] + cells
        print("  ".join(row_parts))
    print(sep)


def main():
    entries: list[tuple[str, str, str]] = []
    for display_name, subdir, model_key in MODELS:
        if model_key == "hybrid":
            entries.append((display_name, model_key, ""))
        else:
            best_path = MODELS_DIR / subdir / "best.pt"
            if not best_path.exists():
                print(f"  [skip] {display_name}: best.pt not found at {best_path}")
                continue
            entries.append((display_name, model_key, str(best_path)))

    jobs = [
        (display_name, model_key, path, task_id, mix)
        for display_name, model_key, path in entries
        for task_id in TASKS
        for mix in (False, True)
    ]
    n_jobs = len(jobs)
    total_steps = n_jobs * T_EVAL

    # results keyed by (display_name, mix)
    results_nomix: dict[str, dict[int, float]] = {n: {} for n, _, _ in entries}
    results_mix:   dict[str, dict[int, float]] = {n: {} for n, _, _ in entries}
    completed = 0

    with mp.Manager() as manager:
        q = manager.Queue()

        with mp.Pool() as pool:
            for display_name, model_key, path, task_id, mix in jobs:
                pool.apply_async(_eval_task, args=(display_name, model_key, path, task_id, mix, q))
            pool.close()

            with tqdm(total=total_steps, desc="Evaluating", unit="step", unit_scale=True) as pbar:
                while completed < n_jobs:
                    msg = q.get()
                    if isinstance(msg, int):
                        pbar.update(msg)
                    else:
                        _, display_name, task_id, mix, avg_alive = msg
                        bucket = results_mix if mix else results_nomix
                        bucket[display_name][task_id] = avg_alive
                        completed += 1
                        pbar.set_postfix(done=f"{completed}/{n_jobs} tasks")

            pool.join()

    model_names = [n for n, _, _ in entries]
    print(f"\nEval conditions:")
    print(f"  models:        {len(entries)}  ({', '.join(model_names)})")
    print(f"  tasks:         {len(TASKS)}  ({', '.join(TASK_NAMES)})")
    print(f"  steps / task:  {T_EVAL:,}")
    print(f"  total steps:   {total_steps:,}")

    _print_table("--- No irrelevant input (nomix) ---", entries, results_nomix)
    _print_table("--- With irrelevant input (mix)  ---", entries, results_mix)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="__array_wrap__ must accept context and return_scalar arguments",
        category=DeprecationWarning,
    )
    main()
