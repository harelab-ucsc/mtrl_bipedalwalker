"""
scripts.rlft.pretrain_config
============================

Typed presets for ``scripts/rlft/pretrain_critic.py`` (the Rudin baseline's
critic-pretraining stage). Pick one with ``--preset``.

``PretrainConfig`` is a frozen dataclass whose field defaults are the base
config; a preset is a ``PretrainConfig(...)`` constructed with its deltas —
direct construction so the editor autocompletes every field. Calling an existing
preset, ``BASE(experiment_name=..., ...)``, derives a sibling with those fields
replaced (a thin ``dataclasses.replace`` wrapper); the rudin / rudin_adv variants
below use this so the plain and adversarial students share a byte-identical
environment, differing only in the two path fields. The dataclass is picklable so
it can be bound into the SubprocVecEnv factory and shipped to worker processes.

This is a *pure-RL* baseline: no behavior cloning, no experts, no adversarial
task sampling. The actor is a frozen distilled student (``load_student_from``, a
bare path under ``MODELS_DIR`` to the student ``.pt``); only the value network
trains. Output goes to ``experiment_name`` under ``MODELS_DIR`` — both are plain
path strings (mirrors scripts/ppo_bc/pretrain_critic_config.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import torch

from mdp.bipedal_walker.student import HIDDEN_BC
from mdp.bipedal_walker.tasks import (
    GAIT,
    ONEHOT,
    SINGLE_TASKS,
    SINGLE_TASKS_GAIT,
    COMBINATION_TASKS_GAIT,
    GaitTask,
    TaskSpec,
)

# Legacy onehot combination recipe (raw 3-bit (walk, flamingo, tilt) rows).
COMBINATION_TASKS: tuple[tuple[int, int, int], ...] = (
    (1, 0, 1),  # walk + tilt
    # (0, 1, 1),  # flamingo + tilt
)


@dataclass(frozen=True)
class PretrainConfig:
    """All hyperparameters for one critic-pretraining run. Field defaults are the base config."""

    # --- identity / source / output (plain paths under MODELS_DIR) ---
    experiment_name: str = ""  # output dir, e.g. "rudin/pretrained_critic/1.0.0"
    load_student_from: str = ""  # distilled student .pt, e.g. "rudin/distill/1.0.0/best.pt"
    timesteps: int = 200 * 1024 * 14

    # obs-bit scheme: must match the distilled student + the finetune stage.
    task_scheme: str = GAIT

    # --- environment ---
    n_train_envs: int = 14
    ep_time: int = 10
    # (vel, tilt) secs between cmd resamples; > ep_time disables that cmd switching.
    cmd_switching_time: tuple[float, float] = (3.0, 4.0)
    # secs between task resamples; > ep_time disables in-episode task switching.
    task_switching_time: float = 11.0
    # When False (default), in-episode task draws are without replacement (no
    # consecutive repeats); errors at init if draws-per-episode > number of tasks.
    task_switch_replacement: bool = False
    cmd_interp_speed: tuple[float, float] = (5.0, 1.0)  # (x_vel, tilt)
    cmd_sample_range: tuple[tuple[float, float], tuple[float, float]] = (
        (-5.0, 5.0),
        (-0.75, 0.75),
    )  # (x_vel, tilt)
    cmd_sample_zero: tuple[float, float] = (0.2, 0.15)
    # Which tasks the critic is pretrained to see, and how often (via switching
    # time) — match this to the finetune stage. Default is the gait combination set.
    allowed_task_mixing: Sequence[
        tuple[int, int, int] | TaskSpec | GaitTask
    ] = COMBINATION_TASKS_GAIT
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
    init_log_std: float = float(np.log(1.0))
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __call__(self, **overrides) -> "PretrainConfig":
        return replace(self, **overrides)


# ---- combination: walk+tilt combos, switching off ----
COMB_200A = PretrainConfig(
    experiment_name="rudin_adv/pretrained_critic/2.0.0c-comb",
    load_student_from="rudin_adv/distill/2.0.0/final.pt",
    task_scheme=GAIT,
    allowed_task_mixing=COMBINATION_TASKS_GAIT,
    task_switching_time=11.0,  # no switching — focus on the combination
    use_indv_task_rew=True,
)
COMB_200 = COMB_200A(
    experiment_name="rudin/pretrained_critic/2.0.0c-comb",
    load_student_from="rudin/distill/2.0.0/final.pt",
)

# ---- switching: 5 single gait tasks, fast switching ----
SWIT_200A = PretrainConfig(
    experiment_name="rudin_adv/pretrained_critic/2.0.0c-swit",
    load_student_from="rudin_adv/distill/2.0.0/final.pt",
    task_scheme=GAIT,
    allowed_task_mixing=SINGLE_TASKS_GAIT,
    task_switching_time=3.0,  # switch fast
    use_indv_task_rew=True,
)
SWIT_200 = SWIT_200A(
    experiment_name="rudin/pretrained_critic/2.0.0c-swit",
    load_student_from="rudin/distill/2.0.0/final.pt",
)

# ---- comb-swit: singles + combos together, fast switching over the union ----
CS_200A = PretrainConfig(
    experiment_name="rudin_adv/pretrained_critic/2.0.0c-cs",
    load_student_from="rudin_adv/distill/2.0.0/final.pt",
    task_scheme=GAIT,
    allowed_task_mixing=(*SINGLE_TASKS_GAIT, *COMBINATION_TASKS_GAIT),
    task_switching_time=3.0,  # switch fast over the union
    use_indv_task_rew=True,
)
CS_200 = CS_200A(
    experiment_name="rudin/pretrained_critic/2.0.0c-cs",
    load_student_from="rudin/distill/2.0.0/final.pt",
)

# --- legacy onehot (1.x.x) ------------------------------------------------------
# Task switching only: pretrain the critic across the 4 directional single tasks.
SWITCHING = PretrainConfig(
    experiment_name="rudin/pretrained_critic/1.0.0",
    task_scheme=ONEHOT,
    load_student_from="rudin/distill/1.0.0/final.pt",
    allowed_task_mixing=SINGLE_TASKS,
)

# Task combination only: walk+tilt / flamingo+tilt (no single-task rows).
COMBINATION = PretrainConfig(
    experiment_name="rudin/pretrained_critic/1.0.0",
    task_scheme=ONEHOT,
    load_student_from="rudin/distill/1.0.0/final.pt",
    allowed_task_mixing=COMBINATION_TASKS,
)

# A mix of single + combination tasks.
MIX = PretrainConfig(
    experiment_name="rudin/pretrained_critic/1.0.0",
    task_scheme=ONEHOT,
    load_student_from="rudin/distill/1.0.0/final.pt",
    allowed_task_mixing=(*SINGLE_TASKS, *COMBINATION_TASKS),
)


# Registry consumed by pretrain_critic.py's --preset flag.
PRESETS: dict[str, PretrainConfig] = {
    # gait (2.x.x) — rudin baselines
    "comb_2.0.0a": COMB_200A,
    "comb_2.0.0": COMB_200,
    "swit_2.0.0a": SWIT_200A,
    "swit_2.0.0": SWIT_200,
    "comb-swit_2.0.0a": CS_200A,
    "comb-swit_2.0.0": CS_200,
    # legacy onehot (1.x.x)
    "switching": SWITCHING,
    "combination": COMBINATION,
    "mix": MIX,
}
