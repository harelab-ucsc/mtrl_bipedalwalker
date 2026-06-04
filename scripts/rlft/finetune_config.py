"""
scripts.rlft.finetune_config
============================

Typed presets for ``scripts/rlft/finetune.py`` (the Rudin baseline's RL
fine-tuning stage). Pick one with ``--preset``.

Kept deliberately separate from ``pretrain_config.py`` so neither file turns
into a monolith. ``FinetuneConfig`` is a frozen, picklable dataclass; field
defaults are the base config and presets are direct constructions of it.

Fine-tuning warm-starts from a pretrained-critic zip (``load_pretrained_from``, a
bare path under ``MODELS_DIR``) and continues in the pure RL env with **stricter
objective clipping and a smaller, decaying LR** than pretraining. The network
architecture is inherited from the loaded zip, so it's not configured here.
Output goes to ``experiment_name`` under ``MODELS_DIR`` — both are plain path
strings (mirrors scripts/ppo_bc/train_config.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
from stable_baselines3.common.utils import LinearSchedule

from mdp.bipedal_walker.tasks import (
    GAIT,
    ONEHOT,
    SINGLE_TASKS,
    SINGLE_TASKS_GAIT,
    COMBINATION_TASKS_GAIT,
    GaitTask,
    TaskSpec,
)

# Legacy onehot combination recipe (3-bit (walk, flamingo, tilt) TaskSpecs).
COMBINATION_TASKS = (
    TaskSpec("walk_forward+tilt", (1, 0, 1), +1),  # forward walk + tilt
    TaskSpec("walk_backward+tilt", (1, 0, 1), -1),  # backward walk + tilt
    # (0, 1, 1),  # flamingo + tilt
)


@dataclass(frozen=True)
class FinetuneConfig:
    """All hyperparameters for one RL fine-tuning run. Field defaults are the base config."""

    # --- identity / source / output (plain paths under MODELS_DIR) ---
    experiment_name: str = ""  # output dir, e.g. "rudin/finetuned/1.0.0"
    load_pretrained_from: str = ""  # pretrained-critic zip, e.g. "rudin/pretrained_critic/1.0.0/final.zip"
    timesteps: int = 200 * 1024 * 14

    # obs-bit scheme: must match the pretrained-critic zip.
    task_scheme: str = GAIT

    # --- environment (switching ON by default — fine-tuning is task-responsive RL) ---
    n_train_envs: int = 14
    n_eval_envs: int = 5
    ep_time: int = 10
    cmd_switching_time: tuple[float, float] = (5.0, 5.0)  # (vel, tilt) secs
    task_switching_time: float = 11.0
    # RLFT does no adversarial task selection, so draws are with replacement.
    task_switch_replacement: bool = True
    cmd_interp_speed: tuple[float, float] = (5.0, 1.0)  # (x_vel, tilt)
    cmd_sample_range: tuple[tuple[float, float], tuple[float, float]] = (
        (-5.0, 5.0),
        (-0.75, 0.75),
    )
    cmd_sample_zero: tuple[float, float] = (0.2, 0.15)
    allowed_task_mixing: Sequence[
        tuple[int, int, int] | TaskSpec | GaitTask
    ] = COMBINATION_TASKS_GAIT
    use_indv_task_rew: bool = True
    hull_x_range: tuple[float, float] = (20.0, 60.0)

    # --- fine-tuning PPO hyperparams (tighter clip, lower decaying LR) ---
    learning_rate: Callable[[float], float] | float = field(
        default_factory=lambda: LinearSchedule(5e-5, 5e-6, 0.8)
    )
    clip_range: Callable[[float], float] | float = 0.1  # stricter than pretrain's 0.2
    n_epochs: int = 15
    n_steps: int = 1024
    batch_size: int = 64
    ent_coef: float = 0.005
    vf_coef: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    # Re-widen the action distribution at the start of fine-tuning so RL can
    # explore beyond the frozen-actor mode the critic was pretrained against.
    init_log_std: float | None = float(np.log(0.5))
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


# --- gait (2.x.x) ---------------------------------------------------------------
# Rudin RLFT: pure RL across the 5 gait single tasks with FAST switching — the
# individual-task reward + rapid switching is the mechanism (no BC here). Loads the
# gait switching pretrained-critic zip.
SWITCHING_GAIT = FinetuneConfig(
    experiment_name="rudin/finetuned/2.0.0",
    task_scheme=GAIT,
    load_pretrained_from="rudin/pretrained_critic/2.0.0/final.zip",
    allowed_task_mixing=SINGLE_TASKS_GAIT,
    task_switching_time=2.0,
)

# Fine-tune on the walk+tilt combination only (gait).
COMBINATION_GAIT = FinetuneConfig(
    experiment_name="rudin/combination/2.0.0",
    task_scheme=GAIT,
    load_pretrained_from="rudin/pretrained_critic_comb/2.0.0/final.zip",
    allowed_task_mixing=COMBINATION_TASKS_GAIT,
)

# --- legacy onehot (1.x.x) ------------------------------------------------------
# Fine-tune across the 4 directional single tasks (task switching).
# SWITCHING = FinetuneConfig(
#     experiment_name="rudin/finetuned/1.0.0",
#     load_pretrained_from="rudin/pretrained_critic/1.0.0/final.zip",
#     task_scheme=ONEHOT,
#     allowed_task_mixing=SINGLE_TASKS,
# )

# Fine-tune on task combinations only.
COMBINATION = FinetuneConfig(
    experiment_name="rudin/combination/1.0.0",
    task_scheme=ONEHOT,
    load_pretrained_from="rudin/pretrained_critic/1.0.0/final.zip",
    allowed_task_mixing=COMBINATION_TASKS,
)

# Fine-tune on a mix of single + combination tasks.
# MIX = FinetuneConfig(
#     experiment_name="rudin/finetuned/1.0.0",
#     load_pretrained_from="rudin/pretrained_critic/1.0.0/final.zip",
#     task_scheme=ONEHOT,
#     allowed_task_mixing=(*SINGLE_TASKS, *COMBINATION_TASKS),
# )


# Registry consumed by finetune.py's --preset flag.
PRESETS: dict[str, FinetuneConfig] = {
    "switching_gait": SWITCHING_GAIT,
    "combination_gait": COMBINATION_GAIT,
    "combination": COMBINATION,
}
