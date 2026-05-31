"""
scripts.ppo_bc.train_config
============================

Typed training presets for ``scripts/ppo_bc/train.py``. Pick one with ``--preset``.

``TrainConfig`` is a frozen dataclass whose field defaults are the base config.
A preset is just a ``TrainConfig(...)`` constructed with its deltas — direct
construction (not ``dataclasses.replace``) so the editor autocompletes every
field. The dataclass is picklable so it can be bound into the SubprocVecEnv
factory and shipped to worker processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
from stable_baselines3.common.utils import LinearSchedule

from mdp.bipedal_walker.tasks import SINGLE_TASKS, TaskSpec
from utils.paths import MODELS_DIR


def _default_expert_paths() -> dict[str, object]:
    # "hop_forward" backs the flamingo task; "body_tilt" backs the tilt task.
    return {
        "walk_forward": MODELS_DIR / "experts" / "walk_forward",
        "walk_backward": MODELS_DIR / "experts" / "walk_backward",
        "hop_forward": MODELS_DIR / "experts" / "hop_forward",
        "body_tilt": MODELS_DIR / "experts" / "body_tilt",
    }


@dataclass(frozen=True)
class TrainConfig:
    """All hyperparameters for one PPO_BC run. Field defaults are the base config."""

    # identity
    experiment_name: str = "ppo_bc/base"
    timesteps: int = 200 * 1024 * 14

    # environment
    n_train_envs: int = 14
    n_eval_envs: int = 5
    ep_time: int = 10
    # (vel, tilt) secs between cmd resamples; > ep_time disables cmd switching.
    cmd_switching_time: tuple[float, float] = (11.0, 11.0)
    # secs between task resamples; > ep_time disables in-episode task switching.
    task_switching_time: float = 11.0
    cmd_interp_speed: tuple[float, float] = (5.0, 1.0)  # (x_vel, tilt)
    cmd_sample_range: tuple[tuple[float, float], tuple[float, float]] = (
        (-5.0, 5.0),
        (-0.75, 0.75),
    )  # (x_vel, tilt)
    cmd_sample_zero: tuple[float, float] = (0.2, 0.15)
    # SINGLE_TASKS = 4 directional single tasks; use raw 3-tuples for combos.
    allowed_task_mixing: Sequence[tuple[int, int, int] | TaskSpec] = SINGLE_TASKS
    # false: stability-only RL reward; true: task-responsive. Combos always get task reward.
    use_indv_task_rew: bool = False
    hull_x_range: tuple[float, float] = (20.0, 60.0)

    # DAgger / BC
    task_bits: int = 3  # trailing obs dims that identify the task
    n_proprio: int = 14  # post-ProprioObsWrapper proprioception size
    act_var_floor: float = 0.05  # additive variance bias/floor on the student action dist
    bc_coef: float = 0.20  # weight on the BC loss in the total loss
    bc_batch_size: int = 256
    bc_loss_type: str = "nll"  # "nll" (log-likelihood) or "mse"
    dagger_max_size: int | None = None  # cap on aggregated demos (None disables)
    collect_data: bool = True  # toggle DAgger relabeling + buffer growth

    # adversarial task sampling
    adversarial_ag: bool = True  # difficulty-weighted task sampling (needs SINGLE_TASKS)
    adversarial_eval_steps_per_task: int = 10000  # eval steps per task per rescore
    adversarial_k: float = 0.85  # 1.0 = pure adversarial, 0.0 = uniform

    # resume / warm-start
    load_dataset: str | None = None  # .npz from demo_dataset.dump (None disables)
    load_model: str | None = None  # prior PPO_BC zip, actor+critic (None disables)
    init_log_std: float | None = float(np.log(1.0))  # log_std override after load_model

    # PPO hyperparams
    learning_rate: Callable[[float], float] | float = field(
        default_factory=lambda: LinearSchedule(5e-4, 3e-5, 0.8)
    )
    n_epochs: int = 15
    n_steps: int = 1024
    batch_size: int = 64
    ent_coef: float = 0.002
    vf_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: Callable[[float], float] | float = 0.1
    target_kl: float | None = None
    max_grad_norm: float = 0.5
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # network architecture
    hidden_dims: tuple[int, ...] = (512, 256, 256, 128, 64)
    critic_hidden_dims: tuple[int, ...] = (1024, 512, 512, 256, 256)
    activation_fn: type[torch.nn.Module] = torch.nn.ELU

    # experts
    expert_paths: dict[str, object] = field(default_factory=_default_expert_paths)


# Primarily-IL single-task pretrain: one fixed task+cmd per episode (switching off),
# stability-only RL reward, BC carries imitation, adversarial up-weights hard tasks.
PRETRAIN = TrainConfig(
    experiment_name="ppo_bc/pretrain/1.0.3",
    timesteps=200*1024*14,
    allowed_task_mixing=SINGLE_TASKS,
    use_indv_task_rew=False,
    adversarial_ag=False,
    bc_coef=0.7,
    bc_loss_type="nll",
    act_var_floor=0.2,
    learning_rate=LinearSchedule(5e-4, 3e-5, 0.8),
)

# SCAFFOLD — single-task RL fine-tune: warm-start from pretrain, switching on,
# task rewards on so RL refines what BC couldn't. Fill in the TODOs before using.
FINETUNE = TrainConfig(
    experiment_name="ppo_bc/2.0.0",  # TODO: pick the finetune experiment id
    load_model=None,  # TODO: point at the pretrain best/final checkpoint
    ep_time=15,  # TODO: confirm finetune episode length
    cmd_switching_time=(3.0, 4.0),  # switching ON. TODO: tune cadence
    task_switching_time=6.0,
    use_indv_task_rew=True,
    learning_rate=LinearSchedule(1e-4, 1e-5, 0.8),  # TODO: typically lower than pretrain
)

# SCAFFOLD — combined tasks (e.g. walk+tilt): no single expert, so RL-driven via
# task reward. Adversarial off (it requires allowed == SINGLE_TASKS). Fill TODOs.
COMBINATION = TrainConfig(
    experiment_name="ppo_bc/3.0.0",  # TODO: pick the combination experiment id
    load_model=None,  # TODO: point at the finetune/pretrain checkpoint
    allowed_task_mixing=(  # TODO: choose the combinations to train
        *SINGLE_TASKS,
        (1, 0, 1),  # walk + tilt
        (1, 1, 0),  # walk + flamingo
    ),
    ep_time=15,  # TODO: confirm episode length
    cmd_switching_time=(3.0, 4.0),
    task_switching_time=6.0,
    adversarial_ag=False,
)


# Registry consumed by train.py's --preset flag.
PRESETS: dict[str, TrainConfig] = {
    "pretrain": PRETRAIN,
    "finetune": FINETUNE,
    "combination": COMBINATION,
}
