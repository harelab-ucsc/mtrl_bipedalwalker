"""
scripts.distillation.train_config
==================================

Typed training presets for ``scripts/distillation/train.py``. Pick one with
``--preset``.

``DistillConfig`` is a frozen dataclass whose field defaults are the base config.
A preset is just a ``DistillConfig(...)`` constructed with its deltas — direct
construction (not ``dataclasses.replace``) so the editor autocompletes every
field. Mirrors the pattern in ``scripts/ppo_bc/train_config.py``.

Run naming: ``experiment_name`` is hardcoded per preset (e.g.
``rudin_adv/distill/1.0.0`` for adversarial, ``rudin/distill/1.0.0`` otherwise);
it feeds MODELS_DIR / LOGS_DIR directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mdp.bipedal_walker.student import HIDDEN_BC, StudentModel


def _default_expert_paths() -> list[str]:
    # "hop_forward" backs the flamingo task; "body_tilt" backs the tilt task.
    return [
        "experts/walk_forward",
        "experts/walk_backward",
        "experts/hop_forward",
        "experts/body_tilt",
    ]


def _default_task_names() -> list[str]:
    return ["walk_forward", "walk_backward", "flamingo", "tilt"]


@dataclass(frozen=True)
class DistillConfig:
    """All hyperparameters for one DAgger distillation run. Field defaults are the base config."""

    # identity — hardcoded per preset; dir = MODELS_DIR/LOGS_DIR / experiment_name
    experiment_name: str = "rudin_adv/distill/1.0.0"

    # student network — mirrors the ppo_bc actor trunk + head
    hidden_dims: tuple[int, ...] = HIDDEN_BC

    # DAgger hyperparams
    T: int = 1500  # env steps per iteration
    N: int = 80  # number of DAgger iterations
    n_active: int | None = (
        None  # cap on iters worth of recent demos to sample (None = full dataset)
    )
    epoch: int = 30  # training epochs per iteration
    batch_size: int = 256
    lr: float = 1e-3
    decay: float = 1e-2
    use_scheduler: bool = False  # cosine warm-restart scheduler (off = constant lr)
    sched_restart_iters: int = (
        2  # dagger iters per cosine restart (only if use_scheduler)
    )
    t_eval: int = 2000  # eval steps per task
    ckpt_int: int = 1  # dagger iters between checkpoint saves
    best_int: int = 1  # dagger iters between best-model checks

    # env
    ep_time: int = 7  # training episode length (seconds)
    eval_ep_time: int = 10  # eval episode length (seconds)

    # noise — additive diagonal gaussian on the EXECUTED student action during
    # collection (state coverage). Expert labels are stored clean. std = sqrt(act_var).
    act_var: float = 0.2

    # mix mode — feed random irrelevant commands (tilt during walk, vel during
    # flamingo/tilt, ...) so the student learns to ignore them.
    mix_irrelevant_input: bool = True

    # adversarial task sampling — up-weight the worst-performing tasks
    adversarial_task_select: bool = True
    adversarial_k: float = 0.85  # 1.0 = pure adversarial, 0.0 = uniform

    # experts / tasks
    expert_paths: list[str] = field(default_factory=_default_expert_paths)
    task_names: list[str] = field(default_factory=_default_task_names)

    def make_student(self) -> StudentModel:
        return StudentModel(hidden=tuple(self.hidden_dims))


# Base adversarial run with mixed irrelevant inputs
ADVERSARIAL = DistillConfig(
    experiment_name="rudin_adv/distill/1.0.0",
    mix_irrelevant_input=False,
    act_var=0.2
)

# Uniform (non-adversarial) task sampling
BASELINE = DistillConfig(
    experiment_name="rudin/distill/1.0.0",
    adversarial_task_select=False,
    mix_irrelevant_input=False,
    act_var=0.2
)

# Registry consumed by train.py's --preset flag.
PRESETS: dict[str, DistillConfig] = {
    "adversarial": ADVERSARIAL,
    "baseline": BASELINE,
}
