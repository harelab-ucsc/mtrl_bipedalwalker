from dataclasses import dataclass, replace
from typing import Any, Sequence, SupportsFloat

from gymnasium.envs.box2d.bipedal_walker import BipedalWalker
import numpy as np

"""
Compositional finetuning reward for the PPO-BC / Rudin pipeline.

This landscape scores policies that ALREADY carry expert motion priors (from
distillation / BC pretrain). Its job is (1) smooth task switching, (2) reading
sensibly for single AND novel combined tasks (walk+tilt, flamingo+tilt), and
(3) sitting on a scale consistent enough across tasks that the critic stays
well-calibrated. See rl_finetune_rewards.gated.py for the previous (gated)
revision and rl_finetune_rewards.old.py for the original monolithic one.

Design (vs. the old gated form):
  * Tracking is ALWAYS ON with task-conditioned targets instead of gating whole
    terms on/off. The env hands us task-masked commands (cmd_vel/cmd_tilt are 0
    when their task bit is off), so they double as "hold 0" anchors: a walk task
    holds upright (a_target=0) and flamingo/tilt hold station (v_target=0). The
    term set is therefore constant — a task switch only moves (interpolated)
    targets, not the reward's shape or scale.
  * Each tracking channel is a two-tier kernel in [0,1]: a Lorentzian basin
    (heavy polynomial tails -> gradient persists far from the target) plus a
    sharp tanh precision peak (steep pull near zero error, the legacy shape).
  * Channels are combined with weights that sum to 1, so r_task in [0,1] for
    EVERY task (single or combined) — one ceiling, consistent scale.
  * Regularization (smoothness/safety) is bounded and behavior-agnostic. The
    termination penalty stays large and dominant: survival is the precondition
    for any task reward, and being task-agnostic it does not skew cross-task scale.

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

# --- tracking kernel ---------------------------------------------------------
# track(err) = ALPHA * L(err, sigma) + (1 - ALPHA) * F(err, k)  in [0, 1]
#   L(err, s) = s^2 / (err^2 + s^2)   lorentzian kernel
#   F(err, k) = 1 - tanh(k * err^2)   sharp precision peak (legacy k=40 shape).
ALPHA = 0.5
SIGMA_VEL = 1.0  # m/s  — coarse half-width
K_VEL = 40.0  # fine sharpness
SIGMA_ANG = 0.4  # rad  — coarse half-width
K_ANG = 40.0  # fine sharpness

# --- task-channel weights (sum to 1 -> r_task in [0, 1] for any task) --------
W_VEL = 0.45
W_ANG = 0.25  # tilt-following was overpowering the gait signal
W_GAIT = 0.30  # reward maintaining the hop more

# --- gait cadence trace ------------------------------------------------------
# Rhythmic gait (walk/flamingo) fires on sparse contact events; a per-step decay
# spreads each event's credit into a dense [0,1] signal so the gait channel isn't
# ~0 between events (which would make it negligible next to the dense channels).
GAIT_DECAY = 0.95

# --- regularization (bounded; weight * cap keeps each term small) ------------
W_REG_ANG_VEL = -0.1
CAP_ANG_VEL = 1.5
W_REG_JOINT_VEL = -0.05
CAP_JOINT_VEL = 3.0
W_REG_JERK = -0.2
CAP_JERK = 0.75
W_REG_HEIGHT = -0.4
CAP_HEIGHT = 0.5
ALIVE_BONUS = 0.05

TERM_PENALTY = 150.0

# --- behaviour penalty (hop mode only) ---------------------------------------
# Standing on BOTH legs defeats the single-leg hop. Active negative — not just a
# zeroed gait bonus — so the policy can't farm vel+tilt tracking from a stable
# two-legged splay. Fires per-step while both feet are down in hop mode (one-leg
# gait / legacy flamingo); bounded and task-gated so it never skews other tasks.
W_HOP_BOTH_DOWN = -1.0


@dataclass
class RewardState:
    # velocity bookkeeping (consumed by the jerk term; the env wrapper updates
    # these post-step).
    prev_vel_x: float = 0.0
    prev_vel_y: float = 0.0
    prev_accel_x: float = 0.0
    prev_accel_y: float = 0.0
    # leg-contact bookkeeping for the shared gait state machine.
    last_leg_contact: int = -1  # 0 = left, 1 = right, -1 = unset
    last_obs_8: float = 0.0
    last_obs_13: float = 0.0
    steps_since_hop: int = 0
    # decaying cadence-quality trace (see _gait_quality).
    gait_trace: float = 0.0


def _to_pair(
    cfg: list[tuple[str, Any, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    raw = {name: float(r) for name, r, _ in cfg}
    weights = {name: float(w) for name, _, w in cfg}
    return raw, weights


# tracking kernel 

def _lorentzian(err: float, sigma: float) -> float:
    """Heavy-tailed basin in (0, 1]; gradient persists far from the target."""
    return float(sigma * sigma / (err * err + sigma * sigma))


def _fine(err: float, k: float) -> float:
    """Sharp precision peak in (0, 1]; steep gradient near zero error."""
    return float(1.0 - np.tanh(k * err * err))


def track(err: float, sigma: float, k: float) -> float:
    """Two-tier tracking quality in [0, 1]: Lorentzian basin + sharp peak."""
    return ALPHA * _lorentzian(err, sigma) + (1.0 - ALPHA) * _fine(err, k)


# gait state machine

def _leg_contact_event(
    base_obs: np.ndarray,
    last_leg_contact: int,
    last_obs_8: float,
    last_obs_13: float,
    steps_since_hop: int,
) -> tuple[str, int, dict[str, Any]]:
    """Classify the current leg-contact rising edge as a gait event.

    Advanced exactly once per step (regardless of task) so the contact state
    never goes stale across task switches. Returns (event, steps_at_event,
    state_update) where event is "none", "hop" (same leg landed again) or
    "switch" (legs alternated), and steps_at_event is the step count since the
    previous event (used to scale gait-cadence credit).
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


def _gait_quality(
    mode: str,
    event: str,
    steps_at_event: int,
    base_obs: np.ndarray,
    gait_trace: float,
) -> tuple[float, float]:
    """Dense gait-quality reward in [0, 1] selected by locomotion mode.

    Returns (r_gait, new_trace).
      * "walk":  reward leg alternation (a "switch" event); a "hop" event resets it.
      * "hop":   reward same-leg hops (a "hop" event); a "switch" event resets it,
                 and both feet down zeroes the reward (breaks the single-leg hop).
                 Covers directional hops and legacy flamingo (hop in place).
      * "quiet": no locomotion task — reward both feet planted so tilt / idle
                 hold station; the rhythmic trace is irrelevant and reset.

    For the rhythmic modes a "good" event refreshes a trace that decays each step
    (GAIT_DECAY), turning sparse contact events into a dense signal: the agent
    must keep producing good events to keep the reward up. (Note: "hop"/"switch"
    here name leg-contact *events*, distinct from the "hop" locomotion *mode*.)
    """
    both_contact = bool(base_obs[8] == 1 and base_obs[13] == 1)

    if mode == "quiet":
        return (1.0 if both_contact else 0.0), 0.0

    trace = gait_trace * GAIT_DECAY  # smoothing term to make sprase reward more effective
    good, bad = ("switch", "hop") if mode == "walk" else ("hop", "switch")
    if event == good:
        trace = float(np.tanh(steps_at_event / GAIT_SATURATION_STEPS))
    elif event == bad:
        trace = 0.0

    r = trace
    if mode == "hop" and both_contact:
        r = 0.0
    return r, trace


# regularization


def regularization_rew(
    env: BipedalWalker,
    base_obs: np.ndarray,
    terminated: bool,
    a_target: float,
    state: RewardState,
) -> tuple[dict[str, float], dict[str, float]]:
    """Always-on, behavior-agnostic shaping: bounded smoothness/safety penalties,
    a small alive bonus, and the dominant termination penalty. Each penalty is
    clipped so one bad step can't swamp the [0,1] task reward."""
    assert env.hull, "cannot find env.hull — environment may be broken!"

    hull_ang_vel = min(abs(env.hull.angularVelocity) ** 2, CAP_ANG_VEL)
    joint_vel_l2 = min(
        float(
            np.mean(
                [base_obs[5] ** 2, base_obs[7] ** 2, base_obs[10] ** 2, base_obs[12] ** 2]
            )
        ),
        CAP_JOINT_VEL,
    )

    hull_vel_x = env.hull.linearVelocity.x
    hull_vel_y = env.hull.linearVelocity.y
    accel_x = hull_vel_x - state.prev_vel_x
    accel_y = hull_vel_y - state.prev_vel_y
    vel_jerk = min(
        (accel_x - state.prev_accel_x) ** 2 + (accel_y - state.prev_accel_y) ** 2,
        CAP_JERK,
    )

    hull_x = env.hull.position.x
    ground_y = float(np.interp(hull_x, env.terrain_x, env.terrain_y))
    height_above_ground = env.hull.position.y - ground_y
    # Tilt-aware target: leaning geometrically lowers the hull, so scale the
    # upright target by cos(commanded tilt) — don't punish obeying the tilt.
    target_h = TARGET_HEIGHT * float(np.cos(a_target))
    height_err = target_h - height_above_ground
    body_height = min(max(height_err * abs(height_err), 0.0), CAP_HEIGHT)

    termination = 1.0 if terminated else 0.0

    cfg: list[tuple[str, Any, float]] = [
        ("reg_hull_ang_vel", hull_ang_vel, W_REG_ANG_VEL),
        ("reg_joint_vel_l2", joint_vel_l2, W_REG_JOINT_VEL),
        ("reg_vel_jerk", vel_jerk, W_REG_JERK),
        ("reg_body_height", body_height, W_REG_HEIGHT),
        ("alive", 1.0, ALIVE_BONUS),
        ("termination", termination, -TERM_PENALTY),
    ]
    return _to_pair(cfg)


# composition


def compositional_rew(
    env: BipedalWalker,
    base_obs: np.ndarray,
    terminated: bool,
    state: RewardState,
    task_bits: Sequence[int],
    cmd_vel: float,
    cmd_tilt: float,
    mode: str | None = None,
    enable_task_reward: bool = True,
    weight_overrides: dict[str, float] | None = None,
) -> tuple[
    SupportsFloat,
    dict[str, float],
    dict[str, float],
    dict[str, float],
    RewardState,
]:
    """Bounded regularization + (optionally) the normalized task reward.

    Scheme-agnostic: the caller decides the gait mode (via
    ``tasks.reward_mode``) and the conditioned tracking targets, and passes them
    in. This reward only consumes the resulting ``mode`` + commands.

    Args:
        task_bits: the obs's 3 trailing bits. Only used to derive ``mode`` when
            ``mode is None`` (legacy onehot fallback: flamingo > walk > quiet).
        cmd_vel, cmd_tilt: the conditioned tracking targets. Under the gait scheme
            these are the raw commands (the task's ranges already zero whatever is
            irrelevant); under onehot they are the task-masked commands. Either
            way a stationary task tracks v=0 and a non-tilt task tracks angle 0.
        mode: the gait mode to condition on — ``"walk" | "hop" | "quiet"``. When
            None, derived from ``task_bits`` (onehot semantics).
        enable_task_reward: when False (e.g. BC-driven single-task pretrain), only
            the regularization layer applies; the task channels are omitted.
        weight_overrides: optional per-term weight overrides (by component name).

    Returns (total, components, raw, weights, new_state). Each channel/penalty is
    logged as raw * weight so the env's reward_terms / reward_raw / reward_weights
    info keys stay populated.
    """
    assert env.hull, "cannot find env.hull — environment may be broken!"
    if mode is None:  # legacy onehot derivation from (walk, flamingo, tilt) bits
        mode = "hop" if int(task_bits[1]) else ("walk" if int(task_bits[0]) else "quiet")

    # Advance the shared contact state machine every step so bookkeeping never
    # goes stale across task switches (the gated form only advanced it for some
    # tasks, which produced a spurious gait bonus on switching back).
    event, steps_at_event, leg_state_update = _leg_contact_event(
        base_obs,
        state.last_leg_contact,
        state.last_obs_8,
        state.last_obs_13,
        state.steps_since_hop,
    )

    # cmd_tilt is the conditioned hull-angle target (0 when tilt isn't commanded);
    # the regularization height term reads it to stay tilt-aware.
    raw, weights = regularization_rew(env, base_obs, terminated, cmd_tilt, state)

    gait_trace = state.gait_trace
    if enable_task_reward:
        r_vel = track(cmd_vel - env.hull.linearVelocity.x, SIGMA_VEL, K_VEL)
        r_ang = track(cmd_tilt - env.hull.angle, SIGMA_ANG, K_ANG)
        r_gait, gait_trace = _gait_quality(
            mode, event, steps_at_event, base_obs, state.gait_trace
        )

        # Active penalty (not a zeroed channel) for parking on both legs while in
        # the single-leg hop mode; kept out of the track_* namespace so the
        # normalized task ceiling (track_* summing to 1) is preserved.
        both_contact = bool(base_obs[8] == 1 and base_obs[13] == 1)
        hop_both_down = 1.0 if (mode == "hop" and both_contact) else 0.0

        r, w = _to_pair(
            [
                ("track_vel", r_vel, W_VEL),
                ("track_ang", r_ang, W_ANG),
                ("track_gait", r_gait, W_GAIT),
                ("hop_both_down", hop_both_down, W_HOP_BOTH_DOWN),
            ]
        )
        raw.update(r)
        weights.update(w)
    else:
        # decay the trace so it can't carry stale credit if the task reward later
        # re-enables; nothing reads it while disabled.
        gait_trace = state.gait_trace * GAIT_DECAY

    if weight_overrides:
        for k, v in weight_overrides.items():
            if k in weights:
                weights[k] = float(v)

    components = {k: raw[k] * weights[k] for k in raw}
    total = sum(components.values())

    new_state = replace(state, gait_trace=gait_trace, **leg_state_update)
    return total, components, raw, weights, new_state
