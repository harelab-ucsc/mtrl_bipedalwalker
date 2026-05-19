import multiprocessing as mp
import warnings

import numpy as np
import torch
from gymnasium import make
from tqdm import tqdm

from utils.paths import MODELS_DIR
from wrappers.bipedal_walker.distill_env import DistillEnv
from mdp.bipedal_walker.student import (
    StudentModel,
    StudentModelXS,
    StudentModelS,
    StudentModelM,
    StudentModelML,
    StudentModelL,
    StudentModelXL,
    StudentModelXLL,
    StudentModelXLLL,
)
from mdp.bipedal_walker.hybrid import HybridModel

TASK_NAMES = ["walk_forward", "walk_backward", "hop_forward", "hop_backward"]
T_EVAL = 1_000_000
UPDATE_INTERVAL = 500

MODEL_CLASSES = {
    "xs":     StudentModelXS,
    "s":      StudentModelS,
    "m":      StudentModelM,
    "ml":     StudentModelML,
    "l":      StudentModelL,
    "xl":     StudentModelXL,
    "xll":    StudentModelXLL,
    "xlll":   StudentModelXLLL,
    "hybrid": HybridModel,
}

TASKS = list(range(4))


def _configure_env(env: DistillEnv, task_id: int):
    if task_id in (0, 2):  # walk/hop forward
        env.config_hull_reset(x_range=(0.0, 40.0))
        env.config_cmd_vel(sample_range=(0.0, 5.0), interp_time=0.5, switch_time=3, zero_prob=0.2)
    else:  # walk/hop backward
        env.config_hull_reset(x_range=(40.0, 80.0))
        env.config_cmd_vel(sample_range=(-5.0, 0.0), interp_time=0.5, switch_time=3, zero_prob=0.2)


def _eval_task(model_name: str, model_path: str, task_id: int, q) -> None:
    cls = MODEL_CLASSES[model_name]
    model = cls()
    if model_name != "hybrid":
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["policy"])
        model.eval()

    env = DistillEnv(make("BipedalWalker-v3", render_mode=None), ep_time=10)
    _configure_env(env, task_id)

    obs, info = env.reset()
    cmd_vel = info["cmd"]["x_vel"]
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
                done = False
                alive = 0
            else:
                alive += 1

            obs_s = StudentModel.obs(obs, task_id, cmd_vel)
            pred = model.forward(torch.tensor(obs_s, dtype=torch.float32))
            obs, _, term, trunc, info = env.step(pred.numpy())
            cmd_vel = info["cmd"]["x_vel"]
            done = term or trunc

            if (step + 1) % UPDATE_INTERVAL == 0:
                q.put(UPDATE_INTERVAL)
                last_sent = step + 1

    remaining = T_EVAL - last_sent
    if remaining > 0:
        q.put(remaining)

    time_alive.append(alive)
    env.close()
    q.put(("result", model_name, task_id, float(np.mean(time_alive))))


def main():
    entries = [
        (name, str(MODELS_DIR / "distill" / name / "best.pt"))
        for name in MODEL_CLASSES
        if name != "hybrid" and (MODELS_DIR / "distill" / name / "best.pt").exists()
    ]
    entries.append(("hybrid", ""))

    jobs = [(name, path, task_id) for name, path in entries for task_id in TASKS]
    n_jobs = len(jobs)
    total_steps = n_jobs * T_EVAL

    results: dict[str, dict[int, float]] = {name: {} for name, _ in entries}
    completed = 0

    with mp.Manager() as manager:
        q = manager.Queue()

        with mp.Pool() as pool:
            for name, path, task_id in jobs:
                pool.apply_async(_eval_task, args=(name, path, task_id, q))
            pool.close()

            with tqdm(total=total_steps, desc="Evaluating", unit="step", unit_scale=True) as pbar:
                while completed < n_jobs:
                    msg = q.get()
                    if isinstance(msg, int):
                        pbar.update(msg)
                    else:
                        _, model_name, task_id, avg_alive = msg
                        results[model_name][task_id] = avg_alive
                        completed += 1
                        pbar.set_postfix(done=f"{completed}/{n_jobs} tasks")

            pool.join()

    rows = []
    for name, task_map in results.items():
        task_vals = [task_map.get(i) for i in range(len(TASK_NAMES))]
        evaluated = [v for v in task_vals if v is not None]
        avg = float(np.mean(evaluated)) if evaluated else 0.0
        rows.append((name, avg, task_vals))
    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"\nEval conditions:")
    print(f"  models:        {len(entries)}  ({', '.join(n for n, _ in entries)})")
    print(f"  tasks:         {len(TASKS)}  ({', '.join(TASK_NAMES)})")
    print(f"  steps / task:  {T_EVAL:,}")
    print(f"  total steps:   {total_steps:,}")

    name_w = max((len(name) for name, *_ in rows), default=5)
    val_w = 10
    task_col_w = max(len(t) for t in TASK_NAMES)

    header_parts = [f"{'Model':<{name_w}}", f"{'Avg Alive':>{val_w}}"] + [
        f"{t:>{task_col_w}}" for t in TASK_NAMES
    ]
    header = "  ".join(header_parts)
    sep = "=" * len(header)

    print(f"\n{sep}")
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


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="__array_wrap__ must accept context and return_scalar arguments",
        category=DeprecationWarning,
    )
    main()
