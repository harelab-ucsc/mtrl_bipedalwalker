from dataclasses import dataclass, replace
from typing import Any, SupportsFloat

from gymnasium.envs.box2d.bipedal_walker import BipedalWalker
import numpy as np

"""
Proprioceptive observation layout (14 elements):
    [0]       hull_ang
    [1]       hull_ang_vel
    [2]       vel_x             DO NOT FUCKING USE!! These are NORMALIZED!!
    [3]       vel_y             DO NOT FUCKING USE!! These are NORMALIZED!!
    [4]       hip_1_pos
    [5]       hip_1_vel
    [6]       knee_1_pos
    [7]       knee_1_vel
    [8]       leg_1_contact
    [9]       hip_2_pos
    [10]      hip_2_vel
    [11]      knee_2_pos
    [12]      knee_2_vel
    [13]      leg_2_contact
"""

LEG_H = 34 / 30.0
TARGET_HEIGHT = 2 * LEG_H + 0.1
GAIT_SATURATION_STEPS = 30.0


@dataclass
class RewardState:
    # velocity bookkeeping (consumed by velocity_tracking_rew;
    # the env wrapper is responsible for updating these post-step)
    prev_vel_x: float = 0.0
    prev_vel_y: float = 0.0
    prev_accel_x: float = 0.0
    prev_accel_y: float = 0.0
    # leg-contact bookkeeping (consumed and updated by _leg_contact_event,
    # shared by both walk_rew and flamingo_hop_rew)
    last_leg_contact: int = -1  # 0 = left, 1 = right, -1 = unset
    last_obs_8: float = 0.0
    last_obs_13: float = 0.0
    steps_since_hop: int = 0


def _to_pair(
    cfg: list[tuple[str, Any, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    raw = {name: float(r) for name, r, _ in cfg}
    weights = {name: float(w) for name, _, w in cfg}
    return raw, weights


def stability_rew(
    env: BipedalWalker,
    base_obs: np.ndarray,
    terminated: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    """Always-on safety/stability terms: don't thrash, don't fall, don't terminate."""
    assert env.hull, "cannot find env.hull — environment may be broken!"

    hull_ang_vel = abs(env.hull.angularVelocity) ** 2
    joint_vel_l2 = np.mean(
        [base_obs[5] ** 2, base_obs[7] ** 2, base_obs[10] ** 2, base_obs[12] ** 2]
    )

    hull_x = env.hull.position.x
    ground_y = float(np.interp(hull_x, env.terrain_x, env.terrain_y))
    height_above_ground = env.hull.position.y - ground_y
    height_err = TARGET_HEIGHT - height_above_ground
    body_height = max(height_err * abs(height_err), 0)

    termination = 1 if terminated else 0

    cfg: list[tuple[str, Any, float]] = [
        ("hull_ang_vel", hull_ang_vel, -0.1),
        ("joint_vel_l2", joint_vel_l2, -0.05),
        ("body_height", body_height, -0.4),
        ("termination", termination, -150.0),
    ]
    return _to_pair(cfg)


def velocity_tracking_rew(
    env: BipedalWalker,
    cmd_vel: float,
    prev_vel_x: float,
    prev_vel_y: float,
    prev_accel_x: float,
    prev_accel_y: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """X-velocity tracking + jerk minimization."""
    assert env.hull, "cannot find env.hull — environment may be broken!"

    hull_vel_x = env.hull.linearVelocity.x
    hull_vel_y = env.hull.linearVelocity.y

    vel_err = cmd_vel - hull_vel_x
    vel_tracking = vel_err ** 2
    vel_tracking_fine = 1 - np.tanh(40 * vel_tracking)

    accel_x = hull_vel_x - prev_vel_x
    accel_y = hull_vel_y - prev_vel_y
    vel_jerk = (accel_x - prev_accel_x) ** 2 + (accel_y - prev_accel_y) ** 2

    cfg: list[tuple[str, Any, float]] = [
        ("vel_tracking", vel_tracking, -0.3),
        ("vel_tracking_fine", vel_tracking_fine, 1.0),
        ("vel_jerk", vel_jerk, -0.2),
    ]
    return _to_pair(cfg)


def tilt_rew(
    env: BipedalWalker,
    cmd_tilt: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Hull-angle tracking. cmd_tilt=0 is the upright signal; non-zero leans."""
    assert env.hull, "cannot find env.hull — environment may be broken!"

    hull_ang_err = cmd_tilt - env.hull.angle
    hull_ang_tracking = hull_ang_err ** 2
    hull_ang_tracking_fine = 1 - np.tanh(40 * hull_ang_tracking)

    cfg: list[tuple[str, Any, float]] = [
        ("hull_ang_tracking", hull_ang_tracking, -0.3),
        ("hull_ang_tracking_fine", hull_ang_tracking_fine, 1.0),
    ]
    return _to_pair(cfg)


def _leg_contact_event(
    base_obs: np.ndarray,
    last_leg_contact: int,
    last_obs_8: float,
    last_obs_13: float,
    steps_since_hop: int,
) -> tuple[str, int, dict[str, Any]]:
    """Classify the current leg-contact rising edge as a gait event.

    Shared by walk_rew and flamingo_hop_rew so the contact state machine
    advances exactly once per step regardless of which behaviors are active.
    Returns (event, steps_at_event, state_update) where event is "none",
    "hop" (same leg landed again) or "switch" (legs alternated), and
    steps_at_event is the step count since the previous event (used to scale
    gait-cadence rewards).
    """
    event = "none"
    steps_at_event = steps_since_hop

    if last_leg_contact == -1:  # ambiguous last
        if base_obs[8]:
            last_leg_contact = 0
        elif base_obs[13]:
            last_leg_contact = 1
    elif last_leg_contact == 0:  # last contact was leg 1
        if base_obs[8] and not last_obs_8:  # leg 1 again
            event = "hop"
            steps_since_hop = -1
        elif base_obs[13] and not last_obs_13:  # leg 2 = alternation
            event = "switch"
            last_leg_contact = 1
            steps_since_hop = -1
    elif last_leg_contact == 1:  # last contact was leg 2
        if base_obs[13] and not last_obs_13:  # leg 2 again
            event = "hop"
            steps_since_hop = -1
        elif base_obs[8] and not last_obs_8:  # leg 1 = alternation
            event = "switch"
            last_leg_contact = 0
            steps_since_hop = -1

    steps_since_hop += 0 if last_leg_contact == -1 else 1

    state_update: dict[str, Any] = {
        "last_leg_contact": int(last_leg_contact),
        "last_obs_8": float(base_obs[8]),
        "last_obs_13": float(base_obs[13]),
        "steps_since_hop": int(steps_since_hop),
    }
    return event, steps_at_event, state_update


def walk_rew(
    event: str,
    steps_at_event: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Bipedal gait shaping: reward alternating leg contacts, penalize hopping
    (same leg twice in a row). Applied alongside velocity tracking. Suppressed
    when flamingo is active so it can't cancel the single-leg hop penalty."""
    leg_alt_bonus = (
        float(np.tanh(steps_at_event / GAIT_SATURATION_STEPS))
        if event == "switch"
        else 0.0
    )
    hopping_penalty = 1 if event == "hop" else 0

    cfg: list[tuple[str, Any, float]] = [
        ("leg_alt_bonus", leg_alt_bonus, 1.0),
        ("hopping_penalty", hopping_penalty, -0.3),
    ]
    return _to_pair(cfg)


def flamingo_hop_rew(
    base_obs: np.ndarray,
    event: str,
    steps_at_event: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Single-leg bounding pattern: reward same-leg hops, heavily penalize
    alternating legs so the hop dominates even when walk is commanded too."""
    both_leg_contact = 1 if base_obs[8] == 1 and base_obs[13] == 1 else 0
    hopping_bonus = (
        float(np.tanh(steps_at_event / GAIT_SATURATION_STEPS))
        if event == "hop"
        else 0.0
    )
    leg_alt_penalty = 1 if event == "switch" else 0

    cfg: list[tuple[str, Any, float]] = [
        ("leg_alt_penalty", leg_alt_penalty, -3.0),
        ("hopping_bonus", hopping_bonus, 1.0),
        ("both_leg_contact", both_leg_contact, -0.5),
    ]
    return _to_pair(cfg)


def compositional_rew(
    env: BipedalWalker,
    base_obs: np.ndarray,
    terminated: bool,
    state: RewardState,
    cmd_vel: float | None = None,
    cmd_tilt: float | None = None,
    cmd_flamingo: bool = False,
    weight_overrides: dict[str, float] | None = None,
) -> tuple[
    SupportsFloat,
    dict[str, float],
    dict[str, float],
    dict[str, float],
    RewardState,
]:
    """Sum stability + any active behavior rewards.

    Commands:
        cmd_vel:      None disables velocity tracking; otherwise target x-velocity.
                      Also gates the walk gait reward (leg alternation).
        cmd_tilt:     None disables tilt tracking;     otherwise target hull angle.
        cmd_flamingo: False disables hop reward;       True enables single-leg hopping.
    """
    raw, weights = stability_rew(env, base_obs, terminated)

    if cmd_vel is not None:
        r, w = velocity_tracking_rew(
            env,
            cmd_vel,
            state.prev_vel_x,
            state.prev_vel_y,
            state.prev_accel_x,
            state.prev_accel_y,
        )
        raw.update(r)
        weights.update(w)

    if cmd_tilt is not None:
        r, w = tilt_rew(env, cmd_tilt)
        raw.update(r)
        weights.update(w)

    # Leg-contact gait shaping. Both walk (alternation) and flamingo (hop) read
    # the same contact state machine, so advance it once and pick the behavior:
    # flamingo dominates when commanded (its heavy alternation penalty governs),
    # otherwise the walk alternation reward applies alongside velocity tracking.
    leg_state_update: dict[str, Any] = {}
    if cmd_flamingo or cmd_vel is not None:
        event, steps_at_event, leg_state_update = _leg_contact_event(
            base_obs,
            state.last_leg_contact,
            state.last_obs_8,
            state.last_obs_13,
            state.steps_since_hop,
        )
        if cmd_flamingo:
            r, w = flamingo_hop_rew(base_obs, event, steps_at_event)
        else:
            r, w = walk_rew(event, steps_at_event)
        raw.update(r)
        weights.update(w)

    if weight_overrides:
        for k, v in weight_overrides.items():
            if k in weights:
                weights[k] = float(v)

    components = {k: raw[k] * weights[k] for k in raw}
    total = sum(components.values())

    new_state = replace(state, **leg_state_update) if leg_state_update else state
    return total, components, raw, weights, new_state
