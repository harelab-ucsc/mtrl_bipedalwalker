"""
scripts.ppo_scratch.train_config
================================

Typed presets for ``scripts/ppo_scratch/train.py`` — the from-scratch PPO floor
baseline. Pick one with ``--preset``.

Unlike the ppo_bc / rudin presets, there is **no warm-start and no adversarial
machinery**: a stock ``stable_baselines3.PPO`` is built from random init and
trained one-shot on the full task union. So this config carries the **network
architecture** (the other RL stages inherit it from a loaded zip) and omits
``load_*`` / adversarial fields entirely.

``TrainConfig`` is a frozen, picklable dataclass whose field defaults are the base
config; a preset is a ``TrainConfig(...)`` with its deltas. Calling a preset,
``BASE(experiment_name=...)``, derives a sibling via ``dataclasses.replace``. The
env + PPO + network settings are kept identical to the ppo_bc / rudin comb_switching
RL stage for a fair comparison; only the from-scratch init and the (necessarily
higher) from-scratch LR differ. See the plan / module docstring of train.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
import torch
from stable_baselines3.common.utils import LinearSchedule

from mdp.bipedal_walker.tasks import (
    GAIT,
    SINGLE_TASKS_GAIT,
    COMBINATION_TASKS_GAIT,
    GaitTask,
    TaskSpec,
)


@dataclass(frozen=True)
class TrainConfig:
    """All hyperparameters for one from-scratch PPO baseline run. Field defaults are the base config."""

    # --- identity / output (plain path under MODELS_DIR) ---
    experiment_name: str = ""  # output dir, e.g. "ppo_scratch/comb_switching/2.0.0"
    timesteps: int = 600 * 1024 * 14  # 8.6M = 2x a single ppo_bc/rudin RL-stage run

    # obs-bit scheme: "gait" (default) or legacy "onehot"
    task_scheme: str = GAIT

    # --- environment (mirrors the comb_switching RL stage) ---
    n_train_envs: int = 14
    n_eval_envs: int = 5
    ep_time: int = 10
    cmd_switching_time: tuple[float, float] = (3.0, 4.0)  # (vel, tilt)
    task_switching_time: float = 11.0  # base: off; the preset switches fast (3s)
    task_switch_replacement: bool = False
    cmd_interp_speed: tuple[float, float] = (5.0, 1.0)  # (x_vel, tilt)
    cmd_sample_range: tuple[tuple[float, float], tuple[float, float]] = (
        (-5.0, 5.0),
        (-0.75, 0.75),
    )
    cmd_sample_zero: tuple[float, float] = (0.2, 0.15)
    allowed_task_mixing: Sequence[
        tuple[int, int, int] | TaskSpec | GaitTask
    ] = SINGLE_TASKS_GAIT
    use_indv_task_rew: bool = True  # the same modular reward everyone uses
    hull_x_range: tuple[float, float] = (20.0, 60.0)

    # --- network (same actor + critic as the ppo_bc method) ---
    hidden_dims: tuple[int, ...] = (512, 256, 256, 128, 64)  # actor (pi)
    critic_hidden_dims: tuple[int, ...] = (1024, 512, 512, 256, 256)  # critic (vf)
    activation_fn: type[torch.nn.Module] = torch.nn.ELU
    log_std_init: float = float(np.log(1.0))  # state-independent std init = log(1)

    # --- PPO (stock defaults for from-scratch; see preset comment) ---
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
    clip_range: Callable[[float], float] | float = 0.2
    max_grad_norm: float = 0.5
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __call__(self, **overrides) -> "TrainConfig":
        return replace(self, **overrides)


# Comb-swit: singles + combos together, fast switching over the union — the same
# task distribution as ppo_bc CS_200 / rudin CS_200, but trained from random init in
# one shot. clip_range=0.2 (SB3 default) and ent_coef=0.002 (light entropy) are stock
# PPO defaults, NOT the method's finetune 0.1/0.004. The LR is the from-scratch
# schedule 5e-4->3e-5 (the finetune 5e-5 is for warm-started runs and would not learn
# from scratch). Everything else (env, reward, actor/critic arch) is identical.
CS_200 = TrainConfig(
    experiment_name="ppo_scratch/comb_switching/2.0.0",
    timesteps=600 * 1024 * 14,
    task_scheme=GAIT,
    allowed_task_mixing=(*SINGLE_TASKS_GAIT, *COMBINATION_TASKS_GAIT),
    ep_time=10,
    task_switching_time=3.0,  # switch fast over the union
    use_indv_task_rew=True,
    learning_rate=LinearSchedule(5e-4, 3e-5, 0.8),
    n_epochs=15,
    n_steps=1024,
    batch_size=64,
    ent_coef=0.002,
    vf_coef=0.5,
    clip_range=0.2,
)

# Registry consumed by train.py's --preset flag.
PRESETS: dict[str, TrainConfig] = {
    "comb_switching": CS_200,
}
