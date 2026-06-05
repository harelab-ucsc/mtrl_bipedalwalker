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

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
import torch
from stable_baselines3.common.utils import LinearSchedule

from mdp.bipedal_walker.tasks import (
    GAIT,
    ONE_LEG,
    ONEHOT,
    SINGLE_TASKS,
    SINGLE_TASKS_GAIT,
    COMBINATION_TASKS_GAIT,
    TWO_LEG,
    GaitTask,
    TaskSpec,
)


def _default_expert_paths() -> dict[str, str]:
    # Bare paths under MODELS_DIR; build_experts() in the training script adds the
    # prefix. Under gait, hop_forward/hop_backward back the directional hops and
    # body_tilt backs tilt; under onehot, hop_forward backs flamingo.
    return {
        "walk_forward": "experts/walk_forward",
        "walk_backward": "experts/walk_backward",
        "hop_forward": "experts/hop_forward",
        "hop_backward": "experts/hop_backward",
        "body_tilt": "experts/body_tilt",
    }


@dataclass(frozen=True)
class TrainConfig:
    """All hyperparameters for one PPO_BC run. Field defaults are the base config."""

    # identity
    experiment_name: str = "ppo_bc/base"
    timesteps: int = 300 * 1024 * 14

    # obs-bit scheme: "gait" (default) or "onehot" (legacy). Drives env sampling,
    # expert routing, and which task-mixing recipe `allowed_task_mixing` should use.
    task_scheme: str = GAIT

    # environment config
    n_train_envs: int = 14
    n_eval_envs: int = 5
    ep_time: int = 10  # episode time
    cmd_switching_time: tuple[float, float] = (3.0, 4.0)  # (vel, tilt). These should remain on
    task_switching_time: float = 11.0  # task switching turned off by default
    # When False, in-episode task draws are without replacement (no consecutive
    # repeats); RlFTEnv errors at init if draws-per-episode > number of allowed tasks.
    task_switch_replacement: bool = False
    cmd_interp_speed: tuple[float, float] = (5.0, 1.0)  # (x_vel, tilt)
    cmd_sample_range: tuple[tuple[float, float], tuple[float, float]] = (
        (-5.0, 5.0),
        (-0.75, 0.75),
    )  # (x_vel, tilt)
    cmd_sample_zero: tuple[float, float] = (0.2, 0.15)  # zero command probability
    allowed_task_mixing: Sequence[
        tuple[int, int, int] | TaskSpec | GaitTask
    ] = SINGLE_TASKS_GAIT
    use_indv_task_rew: bool = False
    hull_x_range: tuple[float, float] = (20.0, 60.0)

    # DAgger / BC config
    task_bits: int = 3  # trailing obs dims that identify the task
    n_proprio: int = 14  # post-ProprioObsWrapper proprioception size
    act_var_floor: float = 0  # additive variance bias/floor on the student action dist
    bc_coef: float = 0.20  # weight on the BC loss in the total loss
    bc_batch_size: int = 256
    bc_loss_type: str = "mse"  # nll / mse
    dagger_max_size: int | None = None  # cap on D
    collect_data: bool = True  # whether to collect DAgger rollouts or not

    # adversarial task sampling
    adversarial_ag: bool = True  # difficulty-weighted task sampling over allowed_task_mixing
    adversarial_eval_steps_per_task: int = 5000  # eval steps per task per rescore
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
    expert_paths: dict[str, str] = field(default_factory=_default_expert_paths)
    
    def __call__(self, **overrides) -> "TrainConfig":
        return replace(self, **overrides)


# Primarily-IL single-task pretrain: stability-only RL reward, BC carries imitation,
# adversarial up-weights hard tasks. Task switching is ON (5s < ep_time) and draws
# without replacement, so consecutive segments never repeat the same task.
PRETRAIN = TrainConfig(
    # identity
    experiment_name="ppo_bc_adv/pretrain/1.0.4",
    timesteps=400*1024*14,
    task_scheme=ONEHOT,
    # environment
    allowed_task_mixing=SINGLE_TASKS,
    task_switching_time=5.0,
    task_switch_replacement=False,
    use_indv_task_rew=False,
    # dagger / bc
    act_var_floor=0.2,
    bc_coef=0.2,
    bc_loss_type="mse",
    dagger_max_size=None,
    # adversarial
    adversarial_ag=True,
    adversarial_eval_steps_per_task=5000,
    # ppo
    learning_rate=LinearSchedule(5e-4, 3e-5, 0.8),
)

PRETRAIN_200 = TrainConfig(
    # identity
    experiment_name="ppo_bc/pretrain/2.0.0",
    timesteps=300*1024*14,
    task_scheme=GAIT,
    # environment
    allowed_task_mixing=SINGLE_TASKS_GAIT,
    ep_time=10,
    task_switching_time=11.0,
    task_switch_replacement=False,
    use_indv_task_rew=False,
    # dagger / bc
    act_var_floor=0.2,
    bc_coef=0.5,
    bc_loss_type="mse",
    dagger_max_size=None,
    # adversarial
    adversarial_ag=False,
    adversarial_eval_steps_per_task=0,
    # ppo
    learning_rate=LinearSchedule(5e-4, 3e-5, 0.8),
)

PRETRAIN_300A = TrainConfig(
    # identity
    experiment_name="ppo_bc_adv/pretrain/3.0.0",
    timesteps=600*1024*14,
    task_scheme=GAIT,
    # environment
    allowed_task_mixing=(
        GaitTask("walk_forward", TWO_LEG, (0.0, 5.0), (0.0, 0.0)),
        GaitTask("walk_backward", TWO_LEG, (-5.0, 0.0), (0.0, 0.0)),
        GaitTask("hop_forward", ONE_LEG, (0.0, 5.0), (0.0, 0.0)),
        GaitTask("hop_backward", ONE_LEG, (-5.0, 0.0), (0.0, 0.0)),
        GaitTask("tilt", TWO_LEG, (0.0, 0.0), (-0.75, 0.75)),
        GaitTask("walk_forward+tilt", TWO_LEG, (0.0, 5.0), (-0.75, 0.75)),
        GaitTask("walk_backward+tilt", TWO_LEG, (-5.0, 0.0), (-0.75, 0.75)),
    ),
    ep_time=10,
    task_switching_time=5.0,
    task_switch_replacement=False,
    use_indv_task_rew=False,
    # dagger / bc
    act_var_floor=0.2,
    bc_coef=0.2,
    bc_loss_type="mse",
    dagger_max_size=2_000_000,
    # adversarial — on even though this trains on everything (singles + combos) in
    # one shot. The eval env mirrors allowed_task_mixing and scores every task on
    # time-alive, so the PMF up-weights the hardest tasks (typically the combos).
    adversarial_ag=True,
    adversarial_eval_steps_per_task=5000,
    # ppo
    learning_rate=LinearSchedule(5e-4, 3e-5, 0.8),
)

# Combined tasks (walk+tilt, flamingo+tilt): no single expert, so RL-driven via the
# task reward. The loaded DAgger dataset is single-task expert demos and acts purely as
# a low-weight BC regularizer against forgetting. Adversarial off here by choice —
# only the combos are polled (uniformly); singles are held by the regularizer, not
# retrained.
COMB = TrainConfig(
    # identity
    experiment_name="ppo_bc_adv/combination/1.3.1",
    task_scheme=ONEHOT,
    load_model="ppo_bc_adv/pretrain/1.0.3c/final.zip",  # pretrained critic and actor
    load_dataset="ppo_bc_adv/pretrain/1.0.3.npz",
    # environment
    allowed_task_mixing=(
        TaskSpec("walk_forward+tilt", (1, 0, 1), +1),  # forward walk + tilt
        TaskSpec("walk_backward+tilt", (1, 0, 1), -1),  # backward walk + tilt
        # (0, 1, 1),  # flamingo + tilt
    ),
    ep_time=10,
    cmd_switching_time=(5.0, 5.0),
    task_switching_time=11.0,  # no task switching for now, just focus on task combination
    use_indv_task_rew=True,
    # dagger / bc
    bc_coef=0.1,  # regularization instead of dominating signal
    bc_loss_type="mse",  # always MSE, NLL tries to minimize variance too
    collect_data=False,  # no DAgger during RL
    # adversarial
    adversarial_ag=False,
    # std=0.5 on warm-start matches the critic (1.0.2c) calibration, vs the default 1.0.
    init_log_std=float(np.log(0.5)),
    # ppo
    learning_rate=LinearSchedule(5e-5, 5e-6, 0.8),
    ent_coef=0.005,
    clip_range=0.1,
)

COMB_200A = TrainConfig(
    # identity
    experiment_name="ppo_bc_adv/combination/2.0.0",
    timesteps=300*1024*14,
    task_scheme=GAIT,
    load_model="ppo_bc_adv/pretrain/2.0.0c-comb/final.zip",  # pretrained critic and actor
    load_dataset="ppo_bc_adv/pretrain/2.0.0.npz",
    # environment
    allowed_task_mixing=COMBINATION_TASKS_GAIT,
    ep_time=10,
    task_switching_time=11.0,  # no task switching for now, just focus on task combination
    use_indv_task_rew=True,
    # dagger / bc
    bc_coef=0.1,  # regularization instead of dominating signal
    bc_loss_type="mse",
    collect_data=False,  # no DAgger during RL
    # adversarial — adv lineage continues adversarial task sampling into RL.
    # (decoupled from collect_data; the PMF rescore loop runs on its own.)
    adversarial_ag=True,
    # model initialization
    init_log_std=float(np.log(1)),
    # ppo
    learning_rate=LinearSchedule(5e-5, 5e-6, 0.8),
    ent_coef=0.004,
    clip_range=0.1,
)
COMB_200 = COMB_200A(
    experiment_name="ppo_bc/combination/2.0.0",
    load_model="ppo_bc/pretrain/2.0.0c-comb/final.zip",
    load_dataset="ppo_bc/pretrain/2.0.0.npz",
    adversarial_ag=False,  # non-adv lineage: no adversarial task sampling
)
COMB_201A = COMB_200A(
    experiment_name="ppo_bc_adv/combination/2.1.0",
    load_model="ppo_bc_adv/pretrain/2.0.1c-comb/final.zip",
    load_dataset="ppo_bc_adv/pretrain/2.0.1.npz",
)

# Switching: 5 single gait tasks, fast switching. RL counterpart of the c-swit
# critic pretrain — warm-start from that re-fit critic (+ its frozen actor) and
# train on the same fast-switching single-task distribution. Same gait-2.x recipe
# as COMB above; only the env fields differ (singles + 3s switching).
SWIT_200A = TrainConfig(
    # identity
    experiment_name="ppo_bc_adv/switching/2.0.0",
    timesteps=300*1024*14,
    task_scheme=GAIT,
    load_model="ppo_bc_adv/pretrain/2.0.0c-swit/final.zip",  # pretrained critic and actor
    load_dataset="ppo_bc_adv/pretrain/2.0.0.npz",
    # environment
    allowed_task_mixing=SINGLE_TASKS_GAIT,
    ep_time=10,
    task_switching_time=3.0,  # switch fast
    use_indv_task_rew=True,
    # dagger / bc
    bc_coef=0.1,  # regularization instead of dominating signal
    bc_loss_type="mse",
    collect_data=False,  # no DAgger during RL
    # adversarial — adv lineage continues adversarial task sampling into RL.
    # (decoupled from collect_data; the PMF rescore loop runs on its own.)
    adversarial_ag=True,
    # model initialization
    init_log_std=float(np.log(1)),
    # ppo
    learning_rate=LinearSchedule(5e-5, 5e-6, 0.8),
    ent_coef=0.004,
    clip_range=0.1,
)
SWIT_200 = SWIT_200A(
    experiment_name="ppo_bc/switching/2.0.0",
    load_model="ppo_bc/pretrain/2.0.0c-swit/final.zip",
    load_dataset="ppo_bc/pretrain/2.0.0.npz",
    adversarial_ag=False,  # non-adv lineage: no adversarial task sampling
)
SWIT_201A = SWIT_200A(
    experiment_name="ppo_bc_adv/switching/2.1.0",
    load_model="ppo_bc_adv/pretrain/2.0.1c-swit/final.zip",
    load_dataset="ppo_bc_adv/pretrain/2.0.1.npz",
)

# Comb-swit: singles + combos together, fast switching over the union. RL
# counterpart of the c-cs critic pretrain; same recipe as SWIT but the task
# distribution is the union of single + combination gait tasks.
CS_200A = TrainConfig(
    # identity
    experiment_name="ppo_bc_adv/comb_switching/2.0.0",
    timesteps=300*1024*14,
    task_scheme=GAIT,
    load_model="ppo_bc_adv/pretrain/2.0.0c-cs/final.zip",  # pretrained critic and actor
    load_dataset="ppo_bc_adv/pretrain/2.0.0.npz",
    # environment
    allowed_task_mixing=(*SINGLE_TASKS_GAIT, *COMBINATION_TASKS_GAIT),
    ep_time=10,
    task_switching_time=3.0,  # switch fast over the union
    use_indv_task_rew=True,
    # dagger / bc
    bc_coef=0.1,  # regularization instead of dominating signal
    bc_loss_type="mse",
    collect_data=False,  # no DAgger during RL
    # adversarial — adv lineage continues adversarial task sampling into RL.
    # (decoupled from collect_data; the PMF rescore loop runs on its own.)
    adversarial_ag=True,
    # model initialization
    init_log_std=float(np.log(1)),
    # ppo
    learning_rate=LinearSchedule(5e-5, 5e-6, 0.8),
    ent_coef=0.004,
    clip_range=0.1,
)
CS_200 = CS_200A(
    experiment_name="ppo_bc/comb_switching/2.0.0",
    load_model="ppo_bc/pretrain/2.0.0c-cs/final.zip",
    load_dataset="ppo_bc/pretrain/2.0.0.npz",
    adversarial_ag=False,  # non-adv lineage: no adversarial task sampling
)
CS_201A = CS_200A(
    experiment_name="ppo_bc_adv/comb_switching/2.1.0",
    load_model="ppo_bc_adv/pretrain/2.0.1c-cs/final.zip",
    load_dataset="ppo_bc_adv/pretrain/2.0.1.npz",
)

# Registry consumed by train.py's --preset flag.
PRESETS: dict[str, TrainConfig] = {
    # "pretrain": PRETRAIN,
    # "combination": COMBINATION,
    "pretrain_200": PRETRAIN_200,
    "pretrain_300a": PRETRAIN_300A,
    
    "combination_200a": COMB_200A,
    "combination_210a": COMB_201A,
    "combination_200": COMB_200,
    "switching_200a": SWIT_200A,
    "switching_210a": SWIT_201A,
    "switching_200": SWIT_200,
    "comb_switching_200a": CS_200A,
    "comb_switching_210a": CS_201A,
    "comb_switching_200": CS_200,
}
