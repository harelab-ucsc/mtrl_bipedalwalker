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
    StudentModelV2,
    StudentModelXS,
    StudentModelS,
    StudentModelM,
    StudentModelML,
    StudentModelL,
    StudentModelXL,
    StudentModelXLL,
    StudentModelXLLL,
    StudentModelXSV2,
    StudentModelSV2,
    StudentModelMV2,
    StudentModelMLV2,
    StudentModelLV2,
    StudentModelXLV2,
    StudentModelXLLV2,
    StudentModelXLLLV2,
)
from mdp.bipedal_walker.hybrid import HybridModel

TASK_NAMES = ["walk_forward", "walk_backward", "hop_forward", "hop_backward", "body_tilt"]
T_EVAL = 1_000_000
UPDATE_INTERVAL = 500  # steps between progress queue sends

MODEL_CLASSES_V1 = {
    "xs":   StudentModelXS,
    "s":    StudentModelS,
    "m":    StudentModelM,
    "ml":   StudentModelML,
    "l":    StudentModelL,
    "xl":   StudentModelXL,
    "xll":  StudentModelXLL,
    "xlll": StudentModelXLLL,
    "hybrid": HybridModel,
}

MODEL_CLASSES_V2 = {
    "xs":     StudentModelXSV2,
    "s":      StudentModelSV2,
    "m":      StudentModelMV2,
    "ml":     StudentModelMLV2,
    "l":      StudentModelLV2,
    "xl":     StudentModelXLV2,
    "xll":    StudentModelXLLV2,
    "xlll":   StudentModelXLLLV2,
    "hybrid": HybridModel,
}

V1_TASKS = list(range(4))
V2_TASKS = list(range(5))


def _configure_env(env: DistillEnv, task_id: int):
    if task_id == 4:  # body_tilt (V2 only)
        env.config_hull_reset(x_range=(20.0, 60.0))
    elif task_id in (0, 2):
        env.config_hull_reset(x_range=(0.0, 40.0))
        env.config_cmd_vel(sample_range=(0.0, 5.0), interp_time=0.5, switch_time=3, zero_prob=0.2)
    else:
        env.config_hull_reset(x_range=(40.0, 80.0))
        env.config_cmd_vel(sample_range=(-5.0, 0.0), interp_time=0.5, switch_time=3, zero_prob=0.2)


def _eval_task(model_name: str, model_path: str, task_id: int, is_v2: bool, q) -> None:
    # Worker: evaluates one model on one task, streams step-batch updates then a result tuple via q.
    cls = MODEL_CLASSES_V2[model_name] if is_v2 else MODEL_CLASSES_V1[model_name]
    model = cls()
    if model_name != "hybrid":
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["policy"])
        model.eval()

    env = DistillEnv(make("BipedalWalker-v3", render_mode=None), ep_time=10)
    if is_v2:
        env.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=10, interp_time=1, zero_prob=0.15)
    _configure_env(env, task_id)

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

            if is_v2:
                obs_s = StudentModelV2.obs(obs, task_id, cmd_vel, cmd_tilt)
            else:
                obs_s = StudentModel.obs(obs, task_id, cmd_vel)

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
    q.put(("result", model_name, task_id, is_v2, float(np.mean(time_alive))))


def main():
    v1_entries = [
        (name, str(MODELS_DIR / "distill" / name / "best.pt"), False)
        for name in MODEL_CLASSES_V1
        if name != "hybrid" and (MODELS_DIR / "distill" / name / "best.pt").exists()
    ]
    v1_entries.append(("hybrid", "", False))

    v2_entries = [
        (name, str(MODELS_DIR / "distill_v2" / name / "best.pt"), True)
        for name in MODEL_CLASSES_V2
        if (MODELS_DIR / "distill_v2" / name / "best.pt").exists()
    ]
    v2_entries.append(("hybrid", "", True))  # hybrid auto-detects V1/V2 from obs length

    all_entries = v1_entries + v2_entries

    jobs = [
        (name, path, task_id, is_v2)
        for name, path, is_v2 in all_entries
        for task_id in (V2_TASKS if is_v2 else V1_TASKS)
    ]
    n_jobs = len(jobs)
    total_steps = n_jobs * T_EVAL

    # results keyed as (name, is_v2) → {task_id: avg_alive}
    result_key = lambda name, is_v2: f"{name} v2" if is_v2 else name
    results: dict[str, dict[int, float]] = {
        result_key(name, is_v2): {} for name, _, is_v2 in all_entries
    }
    completed = 0

    with mp.Manager() as manager:
        q = manager.Queue()

        with mp.Pool() as pool:
            for name, path, task_id, is_v2 in jobs:
                pool.apply_async(_eval_task, args=(name, path, task_id, is_v2, q))
            pool.close()

            with tqdm(total=total_steps, desc="Evaluating", unit="step", unit_scale=True) as pbar:
                while completed < n_jobs:
                    msg = q.get()
                    if isinstance(msg, int):
                        pbar.update(msg)
                    else:
                        _, model_name, task_id, is_v2, avg_alive = msg
                        results[result_key(model_name, is_v2)][task_id] = avg_alive
                        completed += 1
                        pbar.set_postfix(done=f"{completed}/{n_jobs} tasks")

            pool.join()

    # build sortable rows: (display_name, avg over evaluated tasks, per-task list with None for unevaluated)
    rows = []
    for display_name, task_map in results.items():
        task_vals = [task_map.get(i) for i in range(len(TASK_NAMES))]
        evaluated = [v for v in task_vals if v is not None]
        avg = float(np.mean(evaluated)) if evaluated else 0.0
        rows.append((display_name, avg, task_vals))
    rows.sort(key=lambda r: r[1], reverse=True)

    # eval conditions summary
    n_v1 = len(v1_entries)
    n_v2 = len(v2_entries)
    print(f"\nEval conditions:")
    print(f"  v1 models:       {n_v1}  ({', '.join(n for n, _, _ in v1_entries)})")
    print(f"  v2 models:       {n_v2}  ({', '.join(n for n, _, _ in v2_entries)})")
    print(f"  v1 tasks:        {len(V1_TASKS)}  ({', '.join(TASK_NAMES[i] for i in V1_TASKS)})")
    print(f"  v2 tasks:        {len(V2_TASKS)}  ({', '.join(TASK_NAMES[i] for i in V2_TASKS)})")
    print(f"  steps / task:    {T_EVAL:,}")
    print(f"  total steps:     {total_steps:,}")

    # results table
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
        cells = []
        for v in task_vals:
            if v is None:
                cells.append(f"{'-':>{task_col_w}}")
            else:
                cells.append(f"{v:>{task_col_w}.1f}")
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
