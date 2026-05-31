"""
mdp.bipedal_walker.tasks
========================

Canonical task definitions for the PPO_BC / RlFTEnv setup.

The policy *observation* identifies a task with a 3-bit one-hot
``(walk, flamingo, tilt)``. That representation is intentionally left untouched
here — every trained checkpoint emits/consumes it. What this module adds is the
notion that **walk is two tasks, not one**: ``walk_forward`` and
``walk_backward`` share the same obs bits ``(1, 0, 0)`` but are distinct tasks,
disambiguated by the *sign* of ``cmd_vel`` (``>= 0`` is forward — i.e. a zero
velocity command defaults to forward walk). Each directional task is driven by a
single expert, replacing the old "one walk task that fans out to two experts"
scheme.

Use ``SINGLE_TASKS`` for the four individual tasks, ``resolve_task`` to map an
observation back to its directional task (expert routing), ``constrain_vel_range``
to fold a task's velocity sign into a sampling range, and ``coerce_task`` to wrap
a raw 3-tuple (e.g. a combined task like ``(1, 1, 0)``) as an unconstrained spec.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    """A single task.

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


# --- the four canonical single tasks -------------------------------------------
# Order is significant: it defines the adversarial-PMF / scoring order.
WALK_FORWARD = TaskSpec("walk_forward", (1, 0, 0), +1)
WALK_BACKWARD = TaskSpec("walk_backward", (1, 0, 0), -1)
FLAMINGO = TaskSpec("flamingo", (0, 1, 0), 0)
TILT = TaskSpec("tilt", (0, 0, 1), 0)

SINGLE_TASKS: tuple[TaskSpec, ...] = (WALK_FORWARD, WALK_BACKWARD, FLAMINGO, TILT)

# Lookup by name, for callers that pin a task by string (e.g. eval / forced task).
BY_NAME: dict[str, TaskSpec] = {t.name: t for t in SINGLE_TASKS}


def constrain_vel_range(
    base_range: tuple[float, float], vel_sign: int | None
) -> tuple[float, float]:
    """Fold a task's velocity sign into a sampling range.

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


def resolve_task(task_id_vec, cmd_vel: float) -> TaskSpec | None:
    """Map an observation's task bits + velocity command to a single task.

    Returns the matching directional ``TaskSpec`` for the three single-task
    obs vectors, or ``None`` for combined / unrecognized vectors (which have no
    single expert). ``cmd_vel`` only matters for the walk vector, where its sign
    selects forward (``>= 0``) vs backward (``< 0``).
    """
    t = tuple(int(x) for x in task_id_vec)
    if t == (1, 0, 0):
        return WALK_FORWARD if cmd_vel >= 0 else WALK_BACKWARD
    if t == (0, 1, 0):
        return FLAMINGO
    if t == (0, 0, 1):
        return TILT
    return None


def _name_from_bits(t: tuple[int, int, int]) -> str:
    """Compose a readable name for an arbitrary (possibly combined) task vector."""
    parts = [p for bit, p in zip(t, ("walk", "flamingo", "tilt")) if bit]
    return "+".join(parts) if parts else "idle"


def coerce_task(t) -> TaskSpec:
    """Coerce a raw 3-tuple (or pass through a TaskSpec) to a TaskSpec.

    Raw tuples are treated as **unconstrained** (``vel_sign=None``), preserving
    the pre-refactor symmetric-velocity behavior. Combined tasks (e.g.
    ``(1, 1, 0)``) only ever arrive as raw tuples and stay unconstrained.
    """
    if isinstance(t, TaskSpec):
        return t
    tt = (int(t[0]), int(t[1]), int(t[2]))
    return TaskSpec(_name_from_bits(tt), tt, None)
