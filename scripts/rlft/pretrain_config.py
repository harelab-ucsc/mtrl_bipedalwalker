"""
scripts.rlft.pretrain_config
============================

Typed presets for ``scripts/rlft/pretrain_critic.py`` (the Rudin baseline's
critic-pretraining stage). Pick one with ``--preset``.

``PretrainConfig`` is a frozen dataclass whose field defaults are the base
config; a preset is just a ``PretrainConfig(...)`` with its deltas (direct
construction, not ``dataclasses.replace``, so the editor autocompletes every
field). The dataclass is picklable so it can be bound into the SubprocVecEnv
factory and shipped to worker processes.

This is a *pure-RL* baseline: no behavior cloning, no experts, no adversarial
task sampling. The actor is a frozen distilled student (loaded from
``rudin[_adv]/distill/<distill_version>/best.pt``); only the value network
trains. ``adversarial`` selects the distilled-student *source* (and the output
``rudin_adv`` vs ``rudin`` dir) — task sampling is uniform in both cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from mdp.bipedal_walker.student import HIDDEN_BC
from mdp.bipedal_walker.tasks import SINGLE_TASKS, TaskSpec
from utils.paths import rudin_distill_ckpt, rudin_pretrained_critic_experiment

# Common task-mixing recipes (raw 3-bit (walk, flamingo, tilt) rows / TaskSpecs).
COMBINATION_TASKS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 0),  # walk + flamingo
    (1, 0, 1),  # walk + tilt
)


@dataclass(frozen=True)
class PretrainConfig:
    """All hyperparameters for one critic-pretraining run. Field defaults are the base config."""

    # --- identity / source / output ---
    # adversarial picks the distilled-student source AND the output base dir
    # (rudin_adv vs rudin). distill_version locates the student .pt; version is
    # the output semver under <base>/pretrained_critic/<version>.
    adversarial: bool = False
    distill_version: str = "1.0.0"
    version: str = "1.0.0"
    timesteps: int = 200 * 1024 * 14

    # --- environment ---
    n_train_envs: int = 14
    ep_time: int = 10
    # (vel, tilt) secs between cmd resamples; > ep_time disables that cmd switching.
    cmd_switching_time: tuple[float, float] = (3.0, 4.0)
    # secs between task resamples; > ep_time disables in-episode task switching.
    task_switching_time: float = 6.0
    cmd_interp_speed: tuple[float, float] = (5.0, 1.0)  # (x_vel, tilt)
    cmd_sample_range: tuple[tuple[float, float], tuple[float, float]] = (
        (-5.0, 5.0),
        (-0.75, 0.75),
    )  # (x_vel, tilt)
    cmd_sample_zero: tuple[float, float] = (0.2, 0.15)
    # Which tasks the critic is pretrained to see, and how often (via switching
    # time). SINGLE_TASKS = the 4 directional single tasks (task switching);
    # raw 3-tuples like (1,0,1) add combinations. Mix both for a combined run.
    allowed_task_mixing: Sequence[tuple[int, int, int] | TaskSpec] = SINGLE_TASKS
    # Poll the full modular RLFT reward for individual tasks too (combos always
    # get it). True so the critic sees task-conditioned returns for every task.
    use_indv_task_rew: bool = True
    hull_x_range: tuple[float, float] = (20.0, 60.0)

    # --- network (actor must match the distilled student so weights copy 1:1) ---
    hidden_dims: tuple[int, ...] = HIDDEN_BC
    critic_hidden_dims: tuple[int, ...] = (1024, 512, 512, 256, 256)
    activation_fn: type[torch.nn.Module] = torch.nn.ELU

    # --- pretraining PPO hyperparams (high vf_coef, no entropy bonus, no BC) ---
    learning_rate: float = 1e-3
    n_epochs: int = 20
    n_steps: int = 1024
    batch_size: int = 256
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    max_grad_norm: float = 0.5
    # Tight policy std so the frozen actor stays near the student's mode while
    # the critic learns the return landscape under that policy.
    init_log_std: float = float(np.log(0.1))
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def experiment_name(self) -> str:
        return rudin_pretrained_critic_experiment(self.adversarial, self.version)

    @property
    def student_ckpt(self):
        return rudin_distill_ckpt(self.adversarial, self.distill_version)


# Task switching only: pretrain the critic across the 4 directional single tasks.
SWITCHING = PretrainConfig(
    version="1.0.0",
    allowed_task_mixing=SINGLE_TASKS,
)

# Task combination only: walk+flamingo / walk+tilt (no single-task rows).
COMBINATION = PretrainConfig(
    version="1.0.0",
    allowed_task_mixing=COMBINATION_TASKS,
)

# A mix of single + combination tasks.
MIX = PretrainConfig(
    version="1.0.0",
    allowed_task_mixing=(*SINGLE_TASKS, *COMBINATION_TASKS),
)


# Registry consumed by pretrain_critic.py's --preset flag.
PRESETS: dict[str, PretrainConfig] = {
    "switching": SWITCHING,
    "combination": COMBINATION,
    "mix": MIX,
}
