import multiprocessing as mp
import warnings
from dataclasses import dataclass, field
from typing import Any, SupportsFloat

import numpy as np
import torch
from gymnasium import make
from stable_baselines3 import PPO
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


TASK_NAMES = [
    "walk_forward",
    "walk_backward",
    "flamingo",
    "tilt",
    "walk_forward + flamingo",
    "walk_backward + flamingo"
]
T_EVAL = 1_000_000
UPDATE_INTERVAL = 500
MODEL_CLASSES = {
    "xs":   StudentModelXSV2,
    "s":    StudentModelSV2,
    "m":    StudentModelMV2,
    "ml":   StudentModelMLV2,
    "l":    StudentModelLV2,
    "xl":   StudentModelXLV2,
    "xll":  StudentModelXLLV2,
    "xlll": StudentModelXLLLV2,
}

@dataclass
class ModelSpec:
    name: str
    kind: str  # "student" | "hybrid" | "sb3"
    subdir: str = ""  # student: path under MODELS_DIR
    model_key: str = ""  # student: key into MODEL_CLASSES
    expert_path: str = ""  # sb3: e.g. "experts/walk_forward"
    # sb3: if set, always feed this velocity to the model (flamingo → 0.0)
    pin_cmd_vel: float | None = None
    # if non-empty, only evaluate on these task IDs; others show '-' in the table
    task_filter: list[int] = field(default_factory=list)
    
TEST_TASKS = [0, 1, 2, 3, 4, 5]
MODELS: list[ModelSpec] = [
    # distilled models
    # ModelSpec("ml.1.n02.mix",   "student", subdir="distill_v2/ml.1.n02.mix",   model_key="ml"),
    # ModelSpec("ml.1.n02.nomix", "student", subdir="distill_v2/ml.1.n02.nomix", model_key="ml"),

    ModelSpec("ml.1.1.n00.nomix",  "student", subdir="distill_v2/ml.1.1.n00.nomix",  model_key="ml"),
    ModelSpec("ml.1.1.n00.mix",    "student", subdir="distill_v2/ml.1.1.n00.mix",    model_key="ml"),
    ModelSpec("ml.1.1.n05.nomix",  "student", subdir="distill_v2/ml.1.1.n05.nomix",  model_key="ml"),
    ModelSpec("ml.1.1.n05.mix",    "student", subdir="distill_v2/ml.1.1.n05.mix",    model_key="ml"),

    ModelSpec("ml.1.1.n02.mix",    "student", subdir="distill_v2/ml.1.1.n02.mix",    model_key="ml"),
    ModelSpec("ml.1.1.n02.nomix",  "student", subdir="distill_v2/ml.1.1.n02.nomix",  model_key="ml"),
    ModelSpec("ml.1.1.unif.mix",   "student", subdir="distill_v2/ml.1.1.unif.mix",   model_key="ml"),
    ModelSpec("ml.1.1.unif.nomix", "student", subdir="distill_v2/ml.1.1.unif.nomix", model_key="ml"),
    # hybrid
    ModelSpec("hybrid",            "hybrid"),
    # sb3
    # ModelSpec("walk_fwd [base]",   "sb3", expert_path="experts/walk_forward",  task_filter=[4, 5]),
    # ModelSpec("walk_bwd [base]",   "sb3", expert_path="experts/walk_backward", task_filter=[4, 5]),
    # ModelSpec("flamingo [base]",   "sb3", expert_path="experts/hop_backward",  pin_cmd_vel=0.0),
]

METRIC_LABEL = "Hop Reward"
METRIC_LABEL = "Time Alive"

def compute_step_metric(env: Any, obs: np.ndarray, terminated: bool, cmd_vel: float) -> float:
    # reward, _, _, _ = _compute_hop_rew(env, obs, terminated, cmd_vel)
    reward = 1
    return float(reward)

# ─────────────────────────────────────────────────────────────────────────────


def _compute_hop_rew(
    env: Any, obs: np.ndarray, terminated: bool, cmd_vel: float
) -> tuple[SupportsFloat, dict[str, float], dict[str, float], dict[str, float]]:
    """
    Observation layout (24 elements):
        [0]       hull_ang
        [1]       hull_ang_vel
        [2]       vel_x             DO NOT FUCKING USE!! These are NORMALIZED!!
        [3]       vel_y             DO NOT FUCKING USE!! These are NORMALIZED!!
        [4]       hip_1_pos
        [5]       hip_1_vel
        [6]       knee_1_pos
        [7]       knee_1_vel
        [8]       leg_1_contact
        [9]       hip_2_pos
        [10]      hip_2_vel
        [11]      knee_2_pos
        [12]      knee_2_vel
        [13]      leg_2_contact
        [14:24]   lidar
    """

    hull_vel_x = env.hull.linearVelocity.x
    hull_ang_vel = env.hull.angularVelocity
    hull_ang = env.hull.angle
    hull_x = env.hull.position.x

    vel_err = cmd_vel - hull_vel_x
    vel_tracking = vel_err**2
    vel_tracking_fine = 1 - np.tanh(40 * vel_tracking)
    hull_ang_vel = abs(hull_ang_vel) ** 2
    leg_1_contact = 1 if obs[8] == 1 and obs[13] == 0 else -1
    leg_2_contact = 1 if obs[13] == 1 else 0
    hull_ang_l2 = hull_ang**2
    termination = 1 if terminated else 0
    joint_vel_l2 = (np.mean([obs[5], obs[7], obs[10], obs[12]])) ** 2

    ground_y = float(np.interp(hull_x, env.terrain_x, env.terrain_y))
    height_above_ground = env.hull.position.y - ground_y

    TARGET_HEIGHT = 2 * (34 / 30.0)  # 2 * LEG_H in world units
    body_height = TARGET_HEIGHT - height_above_ground
    hop_bonus = height_above_ground * (1 - np.tanh(5 * abs(vel_err)))
    if obs[8] == 1 or obs[13] == 1 or body_height < -0.15:
        hop_bonus = 0

    rewards_cfg: list[tuple[str, Any, float]] = [
        ("vel_tracking",      vel_tracking,      -0.2),
        ("vel_tracking_fine", vel_tracking_fine,  0.3),
        ("hull_ang_vel",      hull_ang_vel,      -0.1),
        ("leg_1_contact",     leg_1_contact,      0.15),
        ("leg_2_contact",     leg_2_contact,     -1.1),
        ("hop_bonus",         hop_bonus,          0.2),
        ("hull_ang_l2",       hull_ang_l2,       -1.0),
        ("joint_vel_l2",      joint_vel_l2,      -0.02),
        ("body_height",       body_height,       -0.4),
        ("termination",       termination,     -150.0),
    ]

    raw = {name: float(r) for name, r, _ in rewards_cfg}
    weights = {name: float(w) for name, _, w in rewards_cfg}
    components = {name: float(r * w) for name, r, w in rewards_cfg}

    return sum(components.values()), components, raw, weights


def configure_env(e: DistillEnv, task_id: int, mix: bool):
        # config the env to a certain task
        e.set_task(task_id)  # for rendering current task
        
        if task_id == 0:  # walk forward
            x_range = (0.0, 40.0)
            vel_range = (0.0, 5.0)
            e.config_hull_reset(x_range=x_range)
            e.config_cmd_vel(sample_range=vel_range, interp_time=0.5, switch_time=3, zero_prob=0.2)
            
            if mix:  # mix in random tilt commands as well
                e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
            else:  # reset tilt command to 0 for clean input
                e.config_cmd_tilt(zero_prob=1)
                
            active_task_bits = (1, 0, 0)
        elif task_id == 1:  # walk backward
            x_range = (40.0, 80.0)
            vel_range = (-5.0, 0.0)
            e.config_hull_reset(x_range=x_range)
            e.config_cmd_vel(sample_range=vel_range, interp_time=0.5, switch_time=3, zero_prob=0.2)
            
            if mix:  # mix in random tilt commands as well
                e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
            else:  # reset tilt command to 0 for clean input
                e.config_cmd_tilt(zero_prob=1)
                
            active_task_bits = (1, 0, 0)
        elif task_id == 2:  # flamingo
            x_range = (20.0, 60.0)
            e.config_hull_reset(x_range=x_range)
            
            if mix:  # mix in random tilt and velocity commands
                e.config_cmd_vel(sample_range=(-5.0, 5.0), switch_time=3, interp_time=0.5, zero_prob=0.2)
                e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
            else:  # reset tilt and velocity command to 0 for clean input
                e.config_cmd_vel(zero_prob=1)
                e.config_cmd_tilt(zero_prob=1)
            
            active_task_bits = (0, 1, 0)
        elif task_id == 3:  # tilt
            x_range = (20.0, 60.0)
            tilt_range = (-0.75, 0.75)
            e.config_hull_reset(x_range=x_range)
            e.config_cmd_tilt(sample_range=tilt_range, switch_time=3, interp_time=0.5, zero_prob=0.15)
            
            if mix:  # mix in random velocity commands
                e.config_cmd_vel(sample_range=(-5.0, 5.0), switch_time=3, interp_time=0.5, zero_prob=0.2)
            else:  # reset velocity command to 0 for clean input
                e.config_cmd_vel(zero_prob=1)
                
            active_task_bits = (0, 0, 1)
        elif task_id == 4:  # walk forward + flamingo. The first real fucked up one
            x_range = (0.0, 40.0)
            vel_range = (0.0, 5.0)
            e.config_hull_reset(x_range=x_range)
            e.config_cmd_vel(sample_range=vel_range, interp_time=0.5, switch_time=3, zero_prob=0.2)
            
            if mix:  # mix in random tilt commands as well
                e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
            else:  # reset tilt command to 0 for clean input
                e.config_cmd_tilt(zero_prob=1)
            
            active_task_bits = (1, 1, 0)
        elif task_id == 5:  # walk backward + flamingo.
            x_range = (40.0, 80.0)
            vel_range = (-5.0, 0.0)
            e.config_hull_reset(x_range=x_range)
            e.config_cmd_vel(sample_range=vel_range, interp_time=0.5, switch_time=3, zero_prob=0.2)
            
            if mix:  # mix in random tilt commands as well
                e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
            else:  # reset tilt command to 0 for clean input
                e.config_cmd_tilt(zero_prob=1)
                
            active_task_bits = (1, 1, 0)
        
        e.set_active_tasks(list(active_task_bits))
        return active_task_bits


def _eval_task(spec: ModelSpec, task_id: int, mix: bool, q) -> None:
    env = DistillEnv(make("BipedalWalker-v3", render_mode=None), ep_time=10)
    active_task_bits = configure_env(env, task_id, mix)
    raw_env = env.unwrapped

    # load model
    sb3_model = None
    student_model = None

    if spec.kind == "sb3":
        sb3_model = PPO.load(MODELS_DIR / spec.expert_path, env=None, device="cpu")
    elif spec.kind == "hybrid":
        student_model = HybridModelV2()
    else:  # "student"
        cls = MODEL_CLASSES[spec.model_key]
        student_model = cls()
        ckpt = torch.load(MODELS_DIR / spec.subdir / "best.pt", map_location="cpu", weights_only=False)
        student_model.load_state_dict(ckpt["policy"])
        student_model.eval()

    # reset env and get initial cmds
    obs, info = env.reset()
    cmd_vel = info["cmd"]["x_vel"]
    cmd_tilt = info["cmd"]["tilt"]
    done = False

    # collect ep rewards
    ep_rewards: list[float] = []
    ep_reward = 0.0
    last_sent = 0

    with torch.no_grad():
        for step in range(T_EVAL):
            if done:
                ep_rewards.append(ep_reward)
                obs, info = env.reset()
                cmd_vel = info["cmd"]["x_vel"]
                cmd_tilt = info["cmd"]["tilt"]
                done = False
                ep_reward = 0.0

            # build action for models
            if sb3_model is not None:
                # sb3 experts observe [base_obs(14), cmd_vel]. flamingo always gets vel=0
                vel_for_model = spec.pin_cmd_vel if spec.pin_cmd_vel is not None else cmd_vel
                action, _ = sb3_model.predict(
                    np.append(obs[:14], vel_for_model), deterministic=True
                )
                pred = torch.tensor(action)
            else:
                assert student_model is not None
                obs_s = StudentModelV2.obs(
                    obs, task_id, cmd_vel, cmd_tilt,
                    task_bit_override=active_task_bits,
                )
                pred = student_model.forward(torch.tensor(obs_s, dtype=torch.float32))

            obs, _, term, trunc, info = env.step(pred.numpy())
            cmd_vel = info["cmd"]["x_vel"]
            cmd_tilt = info["cmd"]["tilt"]
            done = term or trunc

            ep_reward += compute_step_metric(raw_env, obs, term, cmd_vel)

            if (step + 1) % UPDATE_INTERVAL == 0:
                q.put(UPDATE_INTERVAL)
                last_sent = step + 1

    remaining = T_EVAL - last_sent
    if remaining > 0:
        q.put(remaining)

    ep_rewards.append(ep_reward)
    env.close()
    q.put(("result", spec.name, task_id, mix, float(np.mean(ep_rewards))))


def _print_table(title: str, entries: list[ModelSpec], results: dict) -> None:
    rows = []
    for spec in entries:
        task_map = results[spec.name]
        task_vals = [task_map.get(i) for i in range(len(TASK_NAMES))]
        evaluated = [v for v in task_vals if v is not None]
        avg = float(np.mean(evaluated)) if evaluated else 0.0
        rows.append((spec.name, avg, task_vals))
    rows.sort(key=lambda r: r[1], reverse=True)

    name_w = max((len(name) for name, *_ in rows), default=5)
    val_w = 12
    task_col_w = max(len(t) for t in TASK_NAMES)
    avg_label = f"Avg {METRIC_LABEL}"

    header_parts = [f"{'Model':<{name_w}}", f"{avg_label:>{val_w}}"] + [
        f"{t:>{task_col_w}}" for t in TASK_NAMES
    ]
    header = "  ".join(header_parts)
    sep = "=" * len(header)

    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for name, avg, task_vals in rows:
        cells = [
            f"{'-':>{task_col_w}}" if v is None else f"{v:>{task_col_w}.2f}"
            for v in task_vals
        ]
        row_parts = [f"{name:<{name_w}}", f"{avg:>{val_w}.2f}"] + cells
        print("  ".join(row_parts))
    print(sep)


def main() -> None:
    entries: list[ModelSpec] = []
    for spec in MODELS:
        if spec.kind == "sb3":
            sb3_path = (MODELS_DIR / spec.expert_path).with_suffix(".zip")
            if not sb3_path.exists():
                print(f"  [skip] {spec.name}: model not found at {sb3_path}")
                continue
            entries.append(spec)
        elif spec.kind == "hybrid":
            entries.append(spec)
        else:  # student
            best_path = MODELS_DIR / spec.subdir / "best.pt"
            if not best_path.exists():
                print(f"  [skip] {spec.name}: best.pt not found at {best_path}")
                continue
            entries.append(spec)

    jobs = [
        (spec, task_id, mix)
        for spec in entries
        for task_id in TEST_TASKS
        if not spec.task_filter or task_id in spec.task_filter
        for mix in (False, True)
    ]
    n_jobs = len(jobs)
    total_steps = n_jobs * T_EVAL

    results_nomix: dict[str, dict[int, float]] = {s.name: {} for s in entries}
    results_mix:   dict[str, dict[int, float]] = {s.name: {} for s in entries}
    completed = 0

    with mp.Manager() as manager:
        q = manager.Queue()

        with mp.Pool() as pool:
            for spec, task_id, mix in jobs:
                pool.apply_async(_eval_task, args=(spec, task_id, mix, q))
            pool.close()

            with tqdm(total=total_steps, desc="Evaluating", unit="step", unit_scale=True) as pbar:
                while completed < n_jobs:
                    msg = q.get()
                    if isinstance(msg, int):
                        pbar.update(msg)
                    else:
                        _, name, task_id, mix, value = msg
                        bucket = results_mix if mix else results_nomix
                        bucket[name][task_id] = value
                        completed += 1
                        pbar.set_postfix(done=f"{completed}/{n_jobs} tasks")

            pool.join()

    model_names = [s.name for s in entries]
    task_names = [TASK_NAMES[i] for i in TEST_TASKS]
    print(f"\nEval conditions:")
    print(f"  metric:        {METRIC_LABEL}")
    print(f"  models:        {len(entries)}  ({', '.join(model_names)})")
    print(f"  tasks:         {len(TEST_TASKS)}  ({', '.join(task_names)})")
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
