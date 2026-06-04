"""
scripts.ppo_bc.pretrain_critic_config
======================================

Typed presets for ``scripts/ppo_bc/pretrain_critic.py``. Pick one with ``--preset``.

``PretrainCriticConfig`` is a frozen dataclass whose field defaults are the base
config. A preset is just a ``PretrainCriticConfig(...)`` constructed with its
deltas — direct construction (not ``dataclasses.replace``) so the editor
autocompletes every field. The dataclass is picklable so it can be bound into
the SubprocVecEnv factory and shipped to worker processes.

Critic pretraining loads a trained PPO_BC actor, freezes it, and trains a fresh
critic so the value network is re-fit to a given reward/env landscape before a
``train.py`` run (the task reward in ``rl_finetune_rewards.compositional_rew`` is
a normalized [0,1]-per-task scale, so its return magnitude differs from the
stability-only pretrain — re-fit the critic before any run that turns task
rewards on or trains combos).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from mdp.bipedal_walker.tasks import (
    GAIT,
    ONEHOT,
    SINGLE_TASKS,
    SINGLE_TASKS_GAIT,
    COMBINATION_TASKS_GAIT,
    GaitTask,
    TaskSpec,
)


def _default_expert_paths() -> dict[str, str]:
    # Bare paths under MODELS_DIR; build_experts() in the script adds the prefix.
    # Under gait, hop_forward/hop_backward back the directional hops; under onehot,
    # hop_forward backs flamingo. "body_tilt" backs tilt.
    return {
        "walk_forward": "experts/walk_forward",
        "walk_backward": "experts/walk_backward",
        "hop_forward": "experts/hop_forward",
        "hop_backward": "experts/hop_backward",
        "body_tilt": "experts/body_tilt",
    }


@dataclass(frozen=True)
class PretrainCriticConfig:
    """All hyperparameters for one critic-pretrain run. Field defaults are the base config.

    The actor (policy trunk + action head + log_std) is loaded from ``load_actor_from``
    and frozen; only the value network trains. There is no BC / DAgger / adversarial
    machinery here — the PPO_BC ctor still requires ``experts`` + ``task_bits`` but they
    are never polled (``bc_coef=0``, ``collect_data=False`` are hardcoded in the script).
    """

    # identity
    experiment_name: str = ""
    timesteps: int = 200 * 1024 * 14
    load_actor_from: str = ""

    # obs-bit scheme: must match the actor zip + the downstream RL stage.
    task_scheme: str = GAIT

    # environment
    n_train_envs: int = 14
    ep_time: int = 10
    # (vel, tilt) secs between cmd resamples; > ep_time disables cmd switching.
    cmd_switching_time: tuple[float, float] = (3.0, 4.0)
    # secs between task resamples; > ep_time disables in-episode task switching.
    task_switching_time: float = 6.0
    cmd_interp_speed: tuple[float, float] = (5.0, 1.0)  # (x_vel, tilt)
    cmd_sample_range: tuple[tuple[float, float], tuple[float, float]] = (
        (-5.0, 5.0),
        (-0.75, 0.75),
    )  # (x_vel, tilt)
    cmd_sample_zero: tuple[float, float] = (0.2, 0.15)
    # Default: the gait single + combination tasks. Match this to the RL stage the
    # critic is being re-fit for (switching uses SINGLE_TASKS_GAIT; combination uses
    # COMBINATION_TASKS_GAIT). Legacy onehot presets override with TaskSpec rows.
    allowed_task_mixing: Sequence[
        tuple[int, int, int] | TaskSpec | GaitTask
    ] = (*SINGLE_TASKS_GAIT, *COMBINATION_TASKS_GAIT)
    # true: task-responsive reward; combos always get task reward regardless.
    use_indv_task_rew: bool = True
    hull_x_range: tuple[float, float] = (20.0, 60.0)

    # PPO_BC ctor bits (relabeling is disabled in the script, but the ctor needs these)
    task_bits: int = 3  # trailing obs dims that identify the task
    n_proprio: int = 14  # post-ProprioObsWrapper proprioception size
    act_var_floor: float = 0.0  # additive variance bias/floor on the action dist

    # pretrain PPO hyperparams (high vf_coef, no ent bonus, no BC)
    learning_rate: Callable[[float], float] | float = 1e-3
    n_epochs: int = 20
    n_steps: int = 1024
    batch_size: int = 256
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: Callable[[float], float] | float = 0.2
    max_grad_norm: float = 0.5
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    # log_std the frozen policy is pinned to
    init_log_std: float = float(np.log(0.5))

    # network architecture
    hidden_dims: tuple[int, ...] = (512, 256, 256, 128, 64)
    critic_hidden_dims: tuple[int, ...] = (1024, 512, 512, 256, 256)
    activation_fn: type[torch.nn.Module] = torch.nn.ELU

    # experts (ctor requirement only; not polled when collect_data=False)
    expert_paths: dict[str, str] = field(default_factory=_default_expert_paths)

# =============================== gait (2.x.x) ===============================
# Re-fit the critic on the SAME gait reward + task distribution as the RL stage
# that follows (this is the key correctness requirement). The SWITCHING critic
# sees the 5 single tasks with fast switching (matching SWITCHING_GAIT); the
# COMBINATION critic sees walk+tilt (matching COMBINATION_GAIT). Both load the
# gait PRETRAIN_GAIT actor and write a `...c` zip the RL preset's load_model uses.
SWITCHING_GAIT = PretrainCriticConfig(
    experiment_name="ppo_bc_gait/pretrain/2.0.0c",
    task_scheme=GAIT,
    load_actor_from="ppo_bc_gait/pretrain/2.0.0/final.zip",
    allowed_task_mixing=SINGLE_TASKS_GAIT,
    task_switching_time=2.0,
    use_indv_task_rew=True,
)
COMBINATION_GAIT = PretrainCriticConfig(
    experiment_name="ppo_bc_gait/pretrain_comb/2.0.0c",
    task_scheme=GAIT,
    load_actor_from="ppo_bc_gait/pretrain/2.0.0/final.zip",
    allowed_task_mixing=COMBINATION_TASKS_GAIT,
    task_switching_time=11.0,  # no switching — focus on the combination
    use_indv_task_rew=True,
)

# =============================== legacy onehot (1.x.x) ===============================
TASK_COMB_ONLY_MIX = (
    (1, 0, 1),  # walk + tilt
    # (0, 1, 1),  # flamingo + tilt
)
TASK_SWIT_COMB_MIX = (
    *SINGLE_TASKS,
    (1, 0, 1),  # walk + tilt
    # (0, 1, 1),  # flamingo + tilt
)

# Re-fit a fresh critic against the (combo-inclusive, task-responsive) reward landscape.
TASK_COMB_ONLY_MSE_LONG = PretrainCriticConfig(
    experiment_name="ppo_bc/pretrain/1.0.2c",
    task_scheme=ONEHOT,
    load_actor_from="ppo_bc/pretrain/1.0.2/final.zip",
    allowed_task_mixing = TASK_COMB_ONLY_MIX
)
TASK_COMB_ONLY_MSE_ADV_LONG = PretrainCriticConfig(
    experiment_name="ppo_bc_adv/pretrain/1.0.2c",
    task_scheme=ONEHOT,
    load_actor_from="ppo_bc_adv/pretrain/1.0.2/final.zip",
    allowed_task_mixing = TASK_COMB_ONLY_MIX
)
TASK_COMB_ONLY_NLL = PretrainCriticConfig(
    experiment_name="ppo_bc/pretrain/1.0.3c",
    task_scheme=ONEHOT,
    load_actor_from="ppo_bc/pretrain/1.0.3/final.zip",
    allowed_task_mixing = TASK_COMB_ONLY_MIX
)
TASK_COMB_ONLY_NLL_ADV = PretrainCriticConfig(
    experiment_name="ppo_bc_adv/pretrain/1.0.3c",
    task_scheme=ONEHOT,
    load_actor_from="ppo_bc_adv/pretrain/1.0.3/final.zip",
    allowed_task_mixing = TASK_COMB_ONLY_MIX
)


# Registry consumed by pretrain_critic.py's --preset flag.
PRESETS: dict[str, PretrainCriticConfig] = {
    "switching_gait": SWITCHING_GAIT,
    "combination_gait": COMBINATION_GAIT,
    "task-comb-only_1.0.2": TASK_COMB_ONLY_MSE_LONG,
    "task-comb-only_1.0.2a": TASK_COMB_ONLY_MSE_ADV_LONG,
    "task-comb-only_1.0.3": TASK_COMB_ONLY_NLL,
    "task-comb-only_1.0.3a": TASK_COMB_ONLY_NLL_ADV,
}
