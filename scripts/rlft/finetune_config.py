"""
scripts.rlft.finetune_config
============================

Typed presets for ``scripts/rlft/finetune.py`` (the Rudin baseline's RL
fine-tuning stage). Pick one with ``--preset``.

Kept deliberately separate from ``pretrain_config.py`` so neither file turns
into a monolith. ``FinetuneConfig`` is a frozen, picklable dataclass whose field
defaults are the base config. A preset is a ``FinetuneConfig(...)`` constructed
with its deltas; calling an existing preset, ``BASE(experiment_name=..., ...)``,
derives a sibling with those fields replaced (a thin ``dataclasses.replace``
wrapper). The rudin / rudin_adv variants below use this so the plain and
adversarial lineages share a byte-identical env + PPO setup, differing only in
the paths + ``adversarial_ag`` (mirrors scripts/ppo_bc/train_config.py).

Fine-tuning warm-starts from a pretrained-critic zip (``load_pretrained_from``, a
bare path under ``MODELS_DIR``) and continues in the pure RL env with **stricter
objective clipping and a smaller, decaying LR** than pretraining. The network
architecture is inherited from the loaded zip, so it's not configured here. The
env + PPO settings mirror ppo_bc's 2.x.x RL presets for a fair baseline (the only
method differences are: no BC here, and the adv lineage runs adversarial
difficulty-weighted task sampling during RL). Output goes to ``experiment_name``
under ``MODELS_DIR`` — both are plain path strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    experiment_name: str = ""  # output dir, e.g. "rudin/combination/2.0.0"
    load_pretrained_from: str = ""  # pretrained-critic zip, e.g. "rudin/pretrained_critic/2.0.0c-comb/final.zip"
    timesteps: int = 300 * 1024 * 14  # matches ppo_bc 2.x.x RL presets

    # obs-bit scheme: must match the pretrained-critic zip.
    task_scheme: str = GAIT

    # --- environment (switching ON by default — fine-tuning is task-responsive RL) ---
    n_train_envs: int = 14
    n_eval_envs: int = 5
    ep_time: int = 10
    cmd_switching_time: tuple[float, float] = (3.0, 4.0)  # (vel, tilt) secs
    task_switching_time: float = 11.0
    # When False (default), in-episode task draws are without replacement (no
    # consecutive repeats) for cleaner per-episode task coverage.
    task_switch_replacement: bool = False
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
    ent_coef: float = 0.004  # matches ppo_bc 2.x.x RL presets
    vf_coef: float = 0.5  # matches ppo_bc base (was 1.0)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    # Re-widen the action distribution at the start of fine-tuning so RL can
    # explore beyond the frozen-actor mode the critic was pretrained against.
    init_log_std: float | None = float(np.log(1.0))
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # --- adversarial task sampling (mirrors ppo_bc train_config) ---
    # When True (adv lineage only), difficulty-weighted task sampling over
    # allowed_task_mixing: the policy is scored per-task by time-alive and the
    # hardest tasks get up-weighted. Expert-independent — see the callback in
    # finetune.py. When False, tasks are drawn uniformly.
    adversarial_ag: bool = False
    adversarial_eval_steps_per_task: int = 5000  # eval steps per task per rescore
    adversarial_k: float = 0.85  # 1.0 = pure adversarial, 0.0 = uniform

    def __call__(self, **overrides) -> "FinetuneConfig":
        """Derive a sibling preset: ``BASE(field=...)`` returns a copy with those
        fields replaced (thin ``dataclasses.replace`` wrapper). The rudin /
        rudin_adv variants use this so the plain and adversarial lineages share a
        byte-identical env + PPO setup, differing only in the paths +
        adversarial_ag."""
        return replace(self, **overrides)


# --- gait (2.x.x) — rudin baselines ---------------------------------------------
# Mirror of scripts/ppo_bc/train_config.py's 2.x.x RL presets: identical env + PPO
# settings for a fair baseline, minus all BC/DAgger/expert machinery (pure RL).
# Three tracks (comb/swit/cs) × two distilled students. Each track's adversarial
# preset is spelled out; the plain sibling derives from it, flipping only the two
# paths + adversarial_ag. The rudin_adv variant continues adversarial task
# selection (difficulty-weighted PMF) into RL, exactly like ppo_bc_adv. Loads the
# matching c-<track> pretrained-critic zip produced by pretrain_critic.py.

# ---- combination: walk+tilt combos, switching off ----
COMB_200A = FinetuneConfig(
    experiment_name="rudin_adv/combination/2.0.0",
    load_pretrained_from="rudin_adv/pretrained_critic/2.0.0c-comb/final.zip",
    task_scheme=GAIT,
    allowed_task_mixing=COMBINATION_TASKS_GAIT,
    task_switching_time=11.0,  # no switching — focus on the combination
    use_indv_task_rew=True,
    adversarial_ag=True,  # adv lineage: difficulty-weighted task sampling in RL
)
COMB_200 = COMB_200A(
    experiment_name="rudin/combination/2.0.0",
    load_pretrained_from="rudin/pretrained_critic/2.0.0c-comb/final.zip",
    adversarial_ag=False,  # plain lineage: uniform task sampling
)

# ---- switching: 5 single gait tasks, fast switching ----
SWIT_200A = FinetuneConfig(
    experiment_name="rudin_adv/switching/2.0.0",
    load_pretrained_from="rudin_adv/pretrained_critic/2.0.0c-swit/final.zip",
    task_scheme=GAIT,
    allowed_task_mixing=SINGLE_TASKS_GAIT,
    task_switching_time=3.0,  # switch fast
    use_indv_task_rew=True,
    adversarial_ag=True,
)
SWIT_200 = SWIT_200A(
    experiment_name="rudin/switching/2.0.0",
    load_pretrained_from="rudin/pretrained_critic/2.0.0c-swit/final.zip",
    adversarial_ag=False,
)

# ---- comb-swit: singles + combos together, fast switching over the union ----
CS_200A = FinetuneConfig(
    experiment_name="rudin_adv/comb_switching/2.0.0",
    load_pretrained_from="rudin_adv/pretrained_critic/2.0.0c-cs/final.zip",
    task_scheme=GAIT,
    allowed_task_mixing=(*SINGLE_TASKS_GAIT, *COMBINATION_TASKS_GAIT),
    task_switching_time=3.0,  # switch fast over the union
    use_indv_task_rew=True,
    adversarial_ag=True,
)
CS_200 = CS_200A(
    experiment_name="rudin/comb_switching/2.0.0",
    load_pretrained_from="rudin/pretrained_critic/2.0.0c-cs/final.zip",
    adversarial_ag=False,
)

# --- legacy onehot (1.x.x) ------------------------------------------------------
# Fine-tune across the 4 directional single tasks (task switching).
# SWITCHING = FinetuneConfig(
#     experiment_name="rudin/finetuned/1.0.0",
#     load_pretrained_from="rudin/pretrained_critic/1.0.0/final.zip",
#     task_scheme=ONEHOT,
#     allowed_task_mixing=SINGLE_TASKS,
# )

# Fine-tune on task combinations only. Pins the fields whose base default moved to
# the gait-2.x values above, so this evaluated 1.0.0 artifact stays reproducible.
COMBINATION = FinetuneConfig(
    experiment_name="rudin/combination/1.0.0",
    task_scheme=ONEHOT,
    load_pretrained_from="rudin/pretrained_critic/1.0.0/final.zip",
    allowed_task_mixing=COMBINATION_TASKS,
    timesteps=200 * 1024 * 14,
    ent_coef=0.005,
    vf_coef=1.0,
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
    # gait (2.x.x) — rudin baselines
    "combination_200a": COMB_200A,
    "combination_200": COMB_200,
    "switching_200a": SWIT_200A,
    "switching_200": SWIT_200,
    "comb_switching_200a": CS_200A,
    "comb_switching_200": CS_200,
    # legacy onehot (1.x.x)
    "combination": COMBINATION,
}
