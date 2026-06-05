"""
scripts.ppo_bc.pretrain_critic_config
======================================

Typed presets for ``scripts/ppo_bc/pretrain_critic.py``. Pick one with ``--preset``.

``PretrainCriticConfig`` is a frozen dataclass whose field defaults are the base
config. A preset is a ``PretrainCriticConfig(...)`` constructed with its deltas —
direct construction so the editor autocompletes every field. Calling an existing
preset, ``BASE(experiment_name=..., ...)``, derives a sibling with those fields
replaced (a thin ``dataclasses.replace`` wrapper); the adversarial/non-adversarial
and version variants below use this to avoid re-spelling shared fields. The
dataclass is picklable so it can be bound into the SubprocVecEnv factory and
shipped to worker processes.

Critic pretraining loads a trained PPO_BC actor, freezes it, and trains a fresh
critic so the value network is re-fit to a given reward/env landscape before a
``train.py`` run (the task reward in ``rl_finetune_rewards.compositional_rew`` is
a normalized [0,1]-per-task scale, so its return magnitude differs from the
stability-only pretrain — re-fit the critic before any run that turns task
rewards on or trains combos).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
import torch

from mdp.bipedal_walker.tasks import (
    GAIT,
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
    init_log_std: float = float(np.log(1.0))

    # network architecture
    hidden_dims: tuple[int, ...] = (512, 256, 256, 128, 64)
    critic_hidden_dims: tuple[int, ...] = (1024, 512, 512, 256, 256)
    activation_fn: type[torch.nn.Module] = torch.nn.ELU

    # experts (ctor requirement only; not polled when collect_data=False)
    expert_paths: dict[str, str] = field(default_factory=_default_expert_paths)

    def __call__(self, **overrides) -> "PretrainCriticConfig":
        """Derive a sibling preset: ``BASE(field=...)`` returns a copy with those
        fields replaced (thin ``dataclasses.replace`` wrapper). Lets the
        adversarial/non-adversarial and version variants share one base instead
        of re-spelling every field."""
        return replace(self, **overrides)


# =============================== gait (2.x.x) ===============================
# Re-fit a fresh critic on the SAME gait reward + task distribution as the RL
# stage that follows (the key correctness requirement), then write a `...c` zip
# the RL preset's load_model resumes. Three tracks × three variants:
#   tracks    comb  walk+tilt combos, switching off          (→ COMBINATION_GAIT)
#             swit  5 single gait tasks, fast switching       (→ SWITCHING_GAIT)
#             cs    singles + combos together, fast switching over the union
#   variants  *_200A  adversarial actor   ppo_bc_adv/pretrain/2.0.0
#             *_200   plain actor          ppo_bc/pretrain/2.0.0
#             *_201A  adversarial actor   ppo_bc_adv/pretrain/2.0.1
# Every variant of a track shares one source actor, so the critic output reuses
# the actor's dir + version with a `c-<track>` suffix to keep the three tracks
# from overwriting each other. Each track's `*_200A` is spelled out; the non-A
# and 2.0.1 siblings derive from it (only the two path fields change).

# ---- combination: walk+tilt combos, switching off (matches COMBINATION_GAIT) ----
COMB_200A = PretrainCriticConfig(
    experiment_name="ppo_bc_adv/pretrain/2.0.0c-comb",
    load_actor_from="ppo_bc_adv/pretrain/2.0.0/final.zip",
    task_scheme=GAIT,
    allowed_task_mixing=COMBINATION_TASKS_GAIT,
    task_switching_time=11.0,  # no switching — focus on the combination
    use_indv_task_rew=True,
)
COMB_200 = COMB_200A(
    experiment_name="ppo_bc/pretrain/2.0.0c-comb",
    load_actor_from="ppo_bc/pretrain/2.0.0/final.zip",
)
COMB_201A = COMB_200A(
    experiment_name="ppo_bc_adv/pretrain/2.0.1c-comb",
    load_actor_from="ppo_bc_adv/pretrain/2.0.1/final.zip",
)

# ---- switching: 5 single gait tasks, fast switching (matches SWITCHING_GAIT) ----
SWIT_200A = PretrainCriticConfig(
    experiment_name="ppo_bc_adv/pretrain/2.0.0c-swit",
    load_actor_from="ppo_bc_adv/pretrain/2.0.0/final.zip",
    task_scheme=GAIT,
    allowed_task_mixing=SINGLE_TASKS_GAIT,
    task_switching_time=3.0,  # switch fast
    use_indv_task_rew=True,
)
SWIT_200 = SWIT_200A(
    experiment_name="ppo_bc/pretrain/2.0.0c-swit",
    load_actor_from="ppo_bc/pretrain/2.0.0/final.zip",
)
SWIT_201A = SWIT_200A(
    experiment_name="ppo_bc_adv/pretrain/2.0.1c-swit",
    load_actor_from="ppo_bc_adv/pretrain/2.0.1/final.zip",
)

# ---- comb-swit: singles + combos together, fast switching over the union ----
CS_200A = PretrainCriticConfig(
    experiment_name="ppo_bc_adv/pretrain/2.0.0c-cs",
    load_actor_from="ppo_bc_adv/pretrain/2.0.0/final.zip",
    task_scheme=GAIT,
    allowed_task_mixing=(*SINGLE_TASKS_GAIT, *COMBINATION_TASKS_GAIT),
    task_switching_time=3.0,  # switch fast over the union
    use_indv_task_rew=True,
)
CS_200 = CS_200A(
    experiment_name="ppo_bc/pretrain/2.0.0c-cs",
    load_actor_from="ppo_bc/pretrain/2.0.0/final.zip",
)
CS_201A = CS_200A(
    experiment_name="ppo_bc_adv/pretrain/2.0.1c-cs",
    load_actor_from="ppo_bc_adv/pretrain/2.0.1/final.zip",
)

# Registry consumed by pretrain_critic.py's --preset flag.
PRESETS: dict[str, PretrainCriticConfig] = {
    "comb_2.0.0a": COMB_200A,
    "comb_2.0.0": COMB_200,
    "comb_2.0.1a": COMB_201A,
    "swit_2.0.0a": SWIT_200A,
    "swit_2.0.0": SWIT_200,
    "swit_2.0.1a": SWIT_201A,
    "comb-swit_2.0.0a": CS_200A,
    "comb-swit_2.0.0": CS_200,
    "comb-swit_2.0.1a": CS_201A,
}
