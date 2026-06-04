"""
mdp.bipedal_walker.tasks
========================

Canonical task definitions for the PPO_BC / RlFTEnv setup.

Two observation **schemes** coexist, selected by a ``task_scheme`` string that
each env / algorithm / config threads through. The 19-dim v2 obs layout
(``[base(14), cmd_vel, cmd_tilt, b0, b1, b2]``) is identical for both — only the
*meaning* of the trailing 3 bits differs:

* ``"onehot"`` (legacy, ``1.x.x`` checkpoints): per-task one-hot
  ``(walk, flamingo, tilt)``. ``walk_forward`` / ``walk_backward`` share
  ``(1, 0, 0)`` and are disambiguated by the sign of ``cmd_vel`` (``>= 0`` is
  forward). Commands are task-masked. Driven by ``TaskSpec`` (+ ``vel_sign``).
* ``"gait"`` (default, ``2.x.x``): the bits are a **gait** ``(two_leg, one_leg,
  unused)`` with the 3rd bit always 0. Two-leg ``(1, 0, 0)`` covers walk and
  tilt (and walk+tilt); one-leg ``(0, 1, 0)`` covers hop_forward / hop_backward
  (flamingo = hop @ ``cmd_vel = 0``). A task is fully described by ``GaitTask``
  (gait + command ranges): the env samples each command straight from the task's
  range, so there is **no** command masking / ``vel_sign`` / tilt flag — the
  ranges already zero whatever is irrelevant (walk ⇒ tilt range ``(0, 0)``;
  tilt ⇒ vel range ``(0, 0)``).

Scheme-aware helpers — ``resolve_task`` (expert routing), ``reward_mode`` (gait
mode for the reward), ``effective_cmd`` (command masking) — branch on the scheme
in one place so call sites stay thin. At ``cmd_vel == 0`` both schemes route to
the *forward* expert (0 walk → walk_forward@0, 0 hop → hop_forward@0).
"""

from __future__ import annotations

from dataclasses import dataclass


# scheme selector
ONEHOT = "onehot"  # legacy per-task one-hot (walk, flamingo, tilt)
GAIT = "gait"  # gait bits (two_leg, one_leg, unused); default for 2.x.x


@dataclass(frozen=True)
class TaskSpec:
    """A single task in the legacy ``"onehot"`` scheme.

    Attributes:
        name: Human-readable id, e.g. ``"walk_forward"``.
        task_id_vec: The 3-bit one-hot ``(walk, flamingo, tilt)`` written into
            the observation. ``walk_forward`` and ``walk_backward`` both use
            ``(1, 0, 0)``.
        vel_sign: Constraint on the velocity command for this task.
            ``+1`` forward (cmd_vel >= 0), ``-1`` backward (cmd_vel < 0),
            ``0`` force zero velocity, ``None`` unconstrained (symmetric — used
            for combined tasks that don't map to a single directional expert).
    """

    name: str
    task_id_vec: tuple[int, int, int]
    vel_sign: int | None


# --- the canonical single tasks (names shared across both schemes) -------------
# Order is significant: it defines the adversarial-PMF / scoring order.
WALK_FORWARD = TaskSpec("walk_forward", (1, 0, 0), +1)
WALK_BACKWARD = TaskSpec("walk_backward", (1, 0, 0), -1)
HOP_FORWARD = TaskSpec("hop_forward", (0, 1, 0), +1)
HOP_BACKWARD = TaskSpec("hop_backward", (0, 1, 0), -1)
FLAMINGO = TaskSpec("flamingo", (0, 1, 0), 0)
TILT = TaskSpec("tilt", (0, 0, 1), 0)

SINGLE_TASKS: tuple[TaskSpec, ...] = (WALK_FORWARD, WALK_BACKWARD, FLAMINGO, TILT)

BY_NAME: dict[str, TaskSpec] = {t.name: t for t in SINGLE_TASKS}


# gait scheme
@dataclass(frozen=True)
class GaitTask:
    """A task in the ``"gait"`` scheme: a gait + command sampling ranges.

    Attributes:
        name: directional/identity id, e.g. ``"walk_forward"`` or
            ``"walk_forward+tilt"``. Single-task names match the expert keys.
        gait: ``(two_leg, one_leg)`` — written into the obs as
            ``(two_leg, one_leg, 0)`` (see ``gait_bits`` / ``task_id_vec``).
        cmd_vel_range: uniform sampling range for ``cmd_vel``. ``(0, 0)`` pins it
            to zero (e.g. tilt / hop-in-place when combined with one-leg).
        cmd_tilt_range: uniform sampling range for ``cmd_tilt``. ``(0, 0)`` pins
            it to zero (e.g. walk / hop).
        label: optional human label (eval / render). Defaults to ``name``.
    """

    name: str
    gait: tuple[int, int]
    cmd_vel_range: tuple[float, float]
    cmd_tilt_range: tuple[float, float]
    label: str | None = None

    @property
    def gait_bits(self) -> tuple[int, int, int]:
        return (int(self.gait[0]), int(self.gait[1]), 0)

    # alias so envs can read .task_id_vec uniformly across both schemes.
    @property
    def task_id_vec(self) -> tuple[int, int, int]:
        return self.gait_bits


# Two-leg covers walk + tilt; one-leg covers directional hops. vel/tilt ranges
# follow the env defaults ((-5, 5) vel, (-0.75, 0.75) tilt).
TWO_LEG = (1, 0)
ONE_LEG = (0, 1)

SINGLE_TASKS_GAIT: tuple[GaitTask, ...] = (
    GaitTask("walk_forward", TWO_LEG, (0.0, 5.0), (0.0, 0.0)),
    GaitTask("walk_backward", TWO_LEG, (-5.0, 0.0), (0.0, 0.0)),
    GaitTask("hop_forward", ONE_LEG, (0.0, 5.0), (0.0, 0.0)),
    GaitTask("hop_backward", ONE_LEG, (-5.0, 0.0), (0.0, 0.0)),
    GaitTask("tilt", TWO_LEG, (0.0, 0.0), (-0.75, 0.75)),
)

COMBINATION_TASKS_GAIT: tuple[GaitTask, ...] = (
    GaitTask("walk_forward+tilt", TWO_LEG, (0.0, 5.0), (-0.75, 0.75)),
    GaitTask("walk_backward+tilt", TWO_LEG, (-5.0, 0.0), (-0.75, 0.75)),
)

BY_NAME_GAIT: dict[str, GaitTask] = {
    t.name: t for t in (*SINGLE_TASKS_GAIT, *COMBINATION_TASKS_GAIT)
}


def constrain_vel_range(
    base_range: tuple[float, float], vel_sign: int | None
) -> tuple[float, float]:
    """Fold a task's velocity sign into a sampling range (onehot scheme).

    ``+1`` clamps to the non-negative half, ``-1`` to the non-positive half,
    ``0`` forces a degenerate ``(0, 0)`` range, and ``None`` returns the range
    unchanged.
    """
    lo, hi = base_range
    if vel_sign is None:
        return (lo, hi)
    if vel_sign > 0:
        return (max(lo, 0.0), max(hi, 0.0))
    if vel_sign < 0:
        return (min(lo, 0.0), min(hi, 0.0))
    return (0.0, 0.0)


def resolve_single_task(task_id_vec, cmd_vel: float, cmd_tilt: float = 0.0, scheme: str = GAIT):
    """Map obs task bits + commands to a single directional task (expert routing).

    Returns a task with a ``.name`` matching the expert keys
    (walk_forward / walk_backward / hop_forward / hop_backward / tilt), or
    ``None`` for combined / unrecognized vectors (which have no single expert).
    At ``cmd_vel == 0`` the *forward* directional task is returned.

    ``scheme="onehot"`` reproduces the legacy routing (``cmd_tilt`` ignored).
    ``scheme="gait"`` reads the bits as ``(two_leg, one_leg, unused)`` and uses
    ``cmd_tilt`` to split walk (tilt == 0) from tilt (tilt != 0); a two-leg
    vector with *both* commands nonzero is a walk+tilt combo → ``None``.
    """
    t = tuple(int(x) for x in task_id_vec)

    if scheme == ONEHOT:
        if t == (1, 0, 0):
            return WALK_FORWARD if cmd_vel >= 0 else WALK_BACKWARD
        if t == (0, 1, 0):
            return FLAMINGO
        if t == (0, 0, 1):
            return TILT
        return None

    # gait scheme
    if t == (0, 1, 0):  # one-leg → hop (flamingo = hop @ 0)
        return HOP_FORWARD if cmd_vel >= 0 else HOP_BACKWARD
    if t == (1, 0, 0):  # two-leg → walk / tilt
        if cmd_vel != 0 and cmd_tilt != 0:
            return None  # walk+tilt combo — no single expert
        if cmd_tilt != 0:
            return TILT
        return WALK_FORWARD if cmd_vel >= 0 else WALK_BACKWARD
    return None


def reward_mode(task_id_vec, cmd_vel: float, scheme: str = GAIT) -> str:
    """Select the gait mode the reward conditions on: ``"walk" | "hop" | "quiet"``.

    ``"onehot"``: flamingo bit → hop, walk bit → walk, else quiet (the legacy
    selection; the same-leg-hop mode is now named ``"hop"``). ``"gait"``: one-leg
    → hop, two-leg with a nonzero velocity → walk, else quiet (a stationary
    two-leg task — tilt / stand — plants its feet).
    """
    t = tuple(int(x) for x in task_id_vec)
    if scheme == ONEHOT:
        if t[1]:
            return "hop"
        if t[0]:
            return "walk"
        return "quiet"
    # gait
    if t[1]:
        return "hop"
    if t[0] and cmd_vel != 0:
        return "walk"
    return "quiet"


def effective_cmd(
    cmd_vec: tuple[float, float], task_id_vec, scheme: str = GAIT
) -> tuple[float, float]:
    """Mask the raw command vector for the policy/reward.

    ``"onehot"``: gate velocity by the walk bit and tilt by the tilt bit (legacy
    behavior). ``"gait"``: identity — nothing is masked, because each task's
    command ranges already zero whatever is irrelevant.
    """
    if scheme == ONEHOT:
        t = tuple(int(x) for x in task_id_vec)
        return (cmd_vec[0] * float(t[0]), cmd_vec[1] * float(t[2]))
    return (cmd_vec[0], cmd_vec[1])


def _name_from_bits(t: tuple[int, int, int]) -> str:
    """Compose a readable name for an arbitrary (possibly combined) task vector."""
    parts = [p for bit, p in zip(t, ("walk", "hop", "tilt")) if bit]
    return "+".join(parts) if parts else "idle"


def coerce_task(t) -> TaskSpec | GaitTask:
    """Coerce a raw 3-tuple to a TaskSpec, or pass a TaskSpec/GaitTask through.

    Raw tuples are treated as **unconstrained** (``vel_sign=None``), preserving
    the pre-refactor symmetric-velocity behavior. Combined tasks (e.g.
    ``(1, 1, 0)``) only ever arrive as raw tuples and stay unconstrained. A
    ``GaitTask`` (gait scheme) passes through untouched.
    """
    if isinstance(t, (TaskSpec, GaitTask)):
        return t
    tt = (int(t[0]), int(t[1]), int(t[2]))
    return TaskSpec(_name_from_bits(tt), tt, None)
