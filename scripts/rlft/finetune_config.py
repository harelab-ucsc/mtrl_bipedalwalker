"""
scripts.rlft.finetune_config
============================

Typed presets for ``scripts/rlft/finetune.py`` (the Rudin baseline's RL
fine-tuning stage). Pick one with ``--preset``.

Kept deliberately separate from ``pretrain_config.py`` so neither file turns
into a monolith. ``FinetuneConfig`` is a frozen, picklable dataclass; field
defaults are the base config and presets are direct constructions of it.

Fine-tuning warm-starts from a pretrained-critic zip and continues in the pure
RL env with **stricter objective clipping and a smaller, decaying LR** than
pretraining. The network architecture is inherited from the loaded zip, so it's
not configured here. ``adversarial`` selects both the pretrained-critic source
and the output dir (rudin_adv vs rudin); task sampling stays uniform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
from stable_baselines3.common.utils import LinearSchedule

from mdp.bipedal_walker.tasks import SINGLE_TASKS, TaskSpec
from utils.paths import (
    MODELS_DIR,
    rudin_finetuned_experiment,
    rudin_pretrained_critic_experiment,
)

# Common task-mixing recipes (raw 3-bit (walk, flamingo, tilt) rows).
COMBINATION_TASKS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 0),  # walk + flamingo
    (1, 0, 1),  # walk + tilt
)


@dataclass(frozen=True)
class FinetuneConfig:
    """All hyperparameters for one RL fine-tuning run. Field defaults are the base config."""

    # --- identity / source / output ---
    # adversarial picks the pretrained-critic source AND the output base dir.
    # pretrained_version locates the input zip; version is the output semver.
    adversarial: bool = False
    pretrained_version: str = "1.0.0"
    version: str = "1.0.0"
    timesteps: int = 200 * 1024 * 14

    # --- environment (switching ON by default — fine-tuning is task-responsive RL) ---
    n_train_envs: int = 14
    n_eval_envs: int = 5
    ep_time: int = 10
    cmd_switching_time: tuple[float, float] = (3.0, 4.0)  # (vel, tilt) secs
    task_switching_time: float = 6.0
    cmd_interp_speed: tuple[float, float] = (5.0, 1.0)  # (x_vel, tilt)
    cmd_sample_range: tuple[tuple[float, float], tuple[float, float]] = (
        (-5.0, 5.0),
        (-0.75, 0.75),
    )
    cmd_sample_zero: tuple[float, float] = (0.2, 0.15)
    allowed_task_mixing: Sequence[tuple[int, int, int] | TaskSpec] = SINGLE_TASKS
    use_indv_task_rew: bool = True
    hull_x_range: tuple[float, float] = (20.0, 60.0)

    # --- fine-tuning PPO hyperparams (tighter clip, lower decaying LR) ---
    learning_rate: Callable[[float], float] | float = field(
        default_factory=lambda: LinearSchedule(2e-5, 8e-6, 0.5)
    )
    clip_range: Callable[[float], float] | float = 0.1  # stricter than pretrain's 0.2
    n_epochs: int = 25
    n_steps: int = 1024
    batch_size: int = 64
    ent_coef: float = 0.006
    vf_coef: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    # Re-widen the action distribution at the start of fine-tuning so RL can
    # explore beyond the frozen-actor mode the critic was pretrained against.
    init_log_std: float | None = float(np.log(1.0))
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def experiment_name(self) -> str:
        return rudin_finetuned_experiment(self.adversarial, self.version)

    @property
    def pretrained_ckpt(self):
        exp = rudin_pretrained_critic_experiment(self.adversarial, self.pretrained_version)
        return MODELS_DIR / exp / "final.zip"


# Fine-tune across the 4 directional single tasks (task switching).
SWITCHING = FinetuneConfig(
    version="1.0.0",
    allowed_task_mixing=SINGLE_TASKS,
)

# Fine-tune on task combinations only.
COMBINATION = FinetuneConfig(
    version="1.0.0",
    allowed_task_mixing=COMBINATION_TASKS,
)

# Fine-tune on a mix of single + combination tasks.
MIX = FinetuneConfig(
    version="1.0.0",
    allowed_task_mixing=(*SINGLE_TASKS, *COMBINATION_TASKS),
)


# Registry consumed by finetune.py's --preset flag.
PRESETS: dict[str, FinetuneConfig] = {
    "switching": SWITCHING,
    "combination": COMBINATION,
    "mix": MIX,
}
