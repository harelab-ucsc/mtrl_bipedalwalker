from typing import Any, SupportsFloat, TypedDict

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

def walk_rew(
    env: BipedalWalker,
    base_obs: np.ndarray,
    cmd_vel: float,
    terminated: bool,
    prev_vel_x: float,
    prev_vel_y: float,
    prev_accel_x: float,
    prev_accel_y: float,
) -> tuple[SupportsFloat, dict[str, float], dict[str, float], dict[str, float]]:
    
    assert env.hull, "cannot find env.hull — environment may be broken!"
    
    hull_vel_x = env.hull.linearVelocity.x
    hull_vel_y = env.hull.linearVelocity.y
    hull_ang_vel = env.hull.angularVelocity
    hull_ang = env.hull.angle
    hull_x = env.hull.position.x

    vel_err = cmd_vel - hull_vel_x
    vel_tracking = vel_err**2
    vel_tracking_fine = 1 - np.tanh(40 * vel_tracking)
    hull_ang_vel = abs(hull_ang_vel) ** 2
    hull_ang_l2 = hull_ang**2
    termination = 1 if terminated else 0
    joint_vel_l2 = np.mean([base_obs[5] ** 2, base_obs[7] ** 2, base_obs[10] ** 2, base_obs[12] ** 2])

    accel_x = hull_vel_x - prev_vel_x
    accel_y = hull_vel_y - prev_vel_y
    vel_jerk = (accel_x - prev_accel_x) ** 2 + (accel_y - prev_accel_y) ** 2

    ground_y = float(np.interp(hull_x, env.terrain_x, env.terrain_y))
    height_above_ground = env.hull.position.y - ground_y

    TARGET_HEIGHT = 2 * (34 / 30.0) + 0.1  # 2 * LEG_H in world units
    height_err = TARGET_HEIGHT - height_above_ground
    body_height = max(height_err * abs(height_err), 0)

    rewards_cfg: list[tuple[str, Any, float]] = [
        ("vel_tracking", vel_tracking, -0.3),
        ("vel_tracking_fine", vel_tracking_fine, 1.0),
        ("hull_ang_vel", hull_ang_vel, -0.1),
        ("hull_ang_l2", hull_ang_l2, -1.0),
        ("joint_vel_l2", joint_vel_l2, -0.1),
        ("body_height", body_height, -0.4),
        ("vel_jerk", vel_jerk, -0.2),
        ("termination", termination, -150.0),
    ]

    raw = {name: float(r) for name, r, _ in rewards_cfg}
    weights = {name: float(w) for name, _, w in rewards_cfg}
    components = {name: float(r * w) for name, r, w in rewards_cfg}

    return sum(components.values()), components, raw, weights

class HopStateUpdate(TypedDict):
    last_leg_contact: int
    last_obs_8: float
    last_obs_13: float
    steps_since_hop: int

def flamingo_rew(
    env: BipedalWalker,
    base_obs: np.ndarray,
    terminated: bool,
    last_leg_contact: int,
    last_obs_8: float,
    last_obs_13: float,
    steps_since_hop: int
) -> tuple[
    SupportsFloat, dict[str, float], dict[str, float], dict[str, float], HopStateUpdate
]:
    
    assert env.hull, "cannot find env.hull — environment may be broken!"
    
    hull_vel_x = env.hull.linearVelocity.x
    hull_ang_vel = env.hull.angularVelocity
    hull_ang = env.hull.angle
    hull_x = env.hull.position.x

    # velocity tracking error
    vel_err = -hull_vel_x
    vel_tracking = vel_err**2
    # fine velocity tracking error
    vel_tracking_fine = 1 - np.tanh(40 * vel_tracking)
    # hull angle velocity
    hull_ang_vel = abs(hull_ang_vel) ** 2
    # both legs on ground simultaneously
    both_leg_contact = 1 if base_obs[8] == 1 and base_obs[13] == 1 else 0
    # hull angle deviation from 0
    hull_ang_l2 = hull_ang**2
    # termination
    termination = 1 if terminated else 0
    # minimize L2 joint_velocity
    joint_vel_l2 = (np.mean([base_obs[5], base_obs[7], base_obs[10], base_obs[12]])) ** 2

    # height above ground (interpolated terrain surface)
    ground_y = float(np.interp(hull_x, env.terrain_x, env.terrain_y))
    height_above_ground = env.hull.position.y - ground_y

    # penalize being close to the ground
    TARGET_HEIGHT = 2 * (34 / 30.0)  # 2 * LEG_H in world units
    body_height = TARGET_HEIGHT - height_above_ground

    # leg alternating penalty + hop bonus
    # penalize if the same leg lands twice in a row (should alternate)
    leg_alt_penalty = 0
    hopping_bonus = 0
    if last_leg_contact == -1:  # ambiguous last
        if base_obs[8]:
            last_leg_contact = 0
        elif base_obs[13]:
            last_leg_contact = 1
    elif last_leg_contact == 0:  # last one is leg 1
        # leg 1 contact on rising edge = reward (same leg lands again after airtime)
        if base_obs[8] and not last_obs_8:
            hopping_bonus = np.tanh(steps_since_hop / 30.0)
            steps_since_hop = -1
        # leg 2 lands = penalty (alternating when we want same-leg bounding)
        elif base_obs[13] and not last_obs_13:
            last_leg_contact = 1
            leg_alt_penalty = 1
            steps_since_hop = -1
    elif last_leg_contact == 1:  # last one is leg 2
        # leg 2 contact on rising edge = reward
        if base_obs[13] and not last_obs_13:
            hopping_bonus = np.tanh(steps_since_hop / 30.0)
            steps_since_hop = -1
        elif base_obs[8] and not last_obs_8:  # leg 1 lands = penalty
            last_leg_contact = 0
            leg_alt_penalty = 1
            steps_since_hop = -1

    # only count when the last state is not ambiguous
    steps_since_hop += 0 if last_leg_contact == -1 else 1

    # update last contact states
    last_obs_8 = base_obs[8]
    last_obs_13 = base_obs[13]

    rewards_cfg: list[tuple[str, Any, float]] = [
        ("vel_tracking", vel_tracking, -0.2),
        ("vel_tracking_fine", vel_tracking_fine, 0.3),
        ("hull_ang_vel", hull_ang_vel, -0.1),
        ("leg_alt_penalty", leg_alt_penalty, -0.3),
        ("hopping_bonus", hopping_bonus, 1.0),
        ("both_leg_contact", both_leg_contact, -0.5),
        ("hull_ang_l2", hull_ang_l2, -1.0),
        ("joint_vel_l2", joint_vel_l2, -0.02),
        ("body_height", body_height, -0.4),
        ("termination", termination, -150.0),
    ]

    raw = {name: float(r) for name, r, _ in rewards_cfg}
    weights = {name: float(w) for name, _, w in rewards_cfg}
    components = {name: float(r * w) for name, r, w in rewards_cfg}
    hop_state_update: HopStateUpdate = {
        "last_leg_contact": last_leg_contact,
        "last_obs_8": last_obs_8,
        "last_obs_13": last_obs_13,
        "steps_since_hop": steps_since_hop,
    }
    
    return sum(components.values()), components, raw, weights, hop_state_update

# combine flamingo reward with walk reward to get a hop reward
def hop_rew(
    env: BipedalWalker,
    base_obs: np.ndarray,
    cmd_vel: float,
    terminated: bool,
    prev_vel_x: float,
    prev_vel_y: float,
    prev_accel_x: float,
    prev_accel_y: float,
    last_leg_contact: int,
    last_obs_8: float,
    last_obs_13: float,
    steps_since_hop: int
) -> tuple[SupportsFloat, dict[str, float], dict[str, float], dict[str, float], HopStateUpdate]:
    
    assert env.hull, "cannot find env.hull — environment may be broken!"
    
    hull_vel_x = env.hull.linearVelocity.x
    hull_vel_y = env.hull.linearVelocity.y
    hull_ang_vel = env.hull.angularVelocity
    hull_ang = env.hull.angle
    hull_x = env.hull.position.x

    vel_err = cmd_vel - hull_vel_x
    vel_tracking = vel_err**2
    vel_tracking_fine = 1 - np.tanh(40 * vel_tracking)
    hull_ang_vel = abs(hull_ang_vel) ** 2
    hull_ang_l2 = hull_ang**2
    termination = 1 if terminated else 0
    joint_vel_l2 = np.mean([base_obs[5] ** 2, base_obs[7] ** 2, base_obs[10] ** 2, base_obs[12] ** 2])
    
    # both legs on ground simultaneously
    both_leg_contact = 1 if base_obs[8] == 1 and base_obs[13] == 1 else 0

    accel_x = hull_vel_x - prev_vel_x
    accel_y = hull_vel_y - prev_vel_y
    vel_jerk = (accel_x - prev_accel_x) ** 2 + (accel_y - prev_accel_y) ** 2

    ground_y = float(np.interp(hull_x, env.terrain_x, env.terrain_y))
    height_above_ground = env.hull.position.y - ground_y

    TARGET_HEIGHT = 2 * (34 / 30.0) + 0.1  # 2 * LEG_H in world units
    height_err = TARGET_HEIGHT - height_above_ground
    body_height = max(height_err * abs(height_err), 0)
    
    # leg alternating penalty + hop bonus
    # penalize if the same leg lands twice in a row (should alternate)
    leg_alt_penalty = 0
    hopping_bonus = 0
    if last_leg_contact == -1:  # ambiguous last
        if base_obs[8]:
            last_leg_contact = 0
        elif base_obs[13]:
            last_leg_contact = 1
    elif last_leg_contact == 0:  # last one is leg 1
        # leg 1 contact on rising edge = reward (same leg lands again after airtime)
        if base_obs[8] and not last_obs_8:
            hopping_bonus = np.tanh(steps_since_hop / 30.0)
            steps_since_hop = -1
        # leg 2 lands = penalty (alternating when we want same-leg bounding)
        elif base_obs[13] and not last_obs_13:
            last_leg_contact = 1
            leg_alt_penalty = 1
            steps_since_hop = -1
    elif last_leg_contact == 1:  # last one is leg 2
        # leg 2 contact on rising edge = reward
        if base_obs[13] and not last_obs_13:
            hopping_bonus = np.tanh(steps_since_hop / 30.0)
            steps_since_hop = -1
        elif base_obs[8] and not last_obs_8:  # leg 1 lands = penalty
            last_leg_contact = 0
            leg_alt_penalty = 1
            steps_since_hop = -1

    # only count when the last state is not ambiguous
    steps_since_hop += 0 if last_leg_contact == -1 else 1

    # update last contact states
    last_obs_8 = base_obs[8]
    last_obs_13 = base_obs[13]

    rewards_cfg: list[tuple[str, Any, float]] = [
        ("vel_tracking", vel_tracking, -0.3),
        ("vel_tracking_fine", vel_tracking_fine, 1.0),
        ("hull_ang_vel", hull_ang_vel, -0.1),
        ("hull_ang_l2", hull_ang_l2, -1.0),
        
        ("leg_alt_penalty", leg_alt_penalty, -0.3),
        ("hopping_bonus", hopping_bonus, 1.0),
        ("both_leg_contact", both_leg_contact, -0.5),
        
        ("joint_vel_l2", joint_vel_l2, -0.1),
        ("body_height", body_height, -0.4),
        ("vel_jerk", vel_jerk, -0.2),
        ("termination", termination, -150.0),
    ]

    raw = {name: float(r) for name, r, _ in rewards_cfg}
    weights = {name: float(w) for name, _, w in rewards_cfg}
    components = {name: float(r * w) for name, r, w in rewards_cfg}
    
    hop_state_update: HopStateUpdate = {
        "last_leg_contact": last_leg_contact,
        "last_obs_8": last_obs_8,
        "last_obs_13": last_obs_13,
        "steps_since_hop": steps_since_hop,
    }

    return sum(components.values()), components, raw, weights, hop_state_update

def tilt_rew(
    env: BipedalWalker,
    base_obs: np.ndarray,
    cmd_tilt: float,
    terminated: bool,
) -> tuple[SupportsFloat, dict[str, float], dict[str, float], dict[str, float]]:

    assert env.hull, "cannot find env.hull — environment may be broken!"

    hull_vel_x = env.hull.linearVelocity.x
    hull_vel_y = env.hull.linearVelocity.y
    hull_ang_vel = env.hull.angularVelocity
    hull_ang = env.hull.angle
    hull_x = env.hull.position.x

    hull_ang_err = cmd_tilt - hull_ang
    hull_ang_tracking = hull_ang_err**2
    hull_ang_tracking_fine = 1 - np.tanh(40 * hull_ang_tracking)
    hull_ang_vel = abs(hull_ang_vel) ** 2
    termination = 1 if terminated else 0
    joint_vel_l2 = np.mean([base_obs[5] ** 2, base_obs[7] ** 2, base_obs[10] ** 2, base_obs[12] ** 2])

    ground_y = float(np.interp(hull_x, env.terrain_x, env.terrain_y))
    height_above_ground = env.hull.position.y - ground_y

    TARGET_HEIGHT = 2.25 * (34 / 30.0)  # 2 * LEG_H — stand tall
    height_err = TARGET_HEIGHT - height_above_ground
    body_height = max(height_err * abs(height_err), 0)

    hull_vel_x_l2 = hull_vel_x**2
    hull_vel_y_l2 = hull_vel_y**2

    rewards_cfg: list[tuple[str, Any, float]] = [
        ("hull_ang_tracking", hull_ang_tracking, -0.3),
        ("hull_ang_tracking_fine", hull_ang_tracking_fine, 1.0),
        ("hull_ang_vel", hull_ang_vel, -0.1),
        ("joint_vel_l2", joint_vel_l2, -0.005),
        ("body_height", body_height, -0.4),
        ("hull_vel_x_l2", hull_vel_x_l2, -0.1),
        ("hull_vel_y_l2", hull_vel_y_l2, -0.07),
        ("termination", termination, -150.0),
    ]

    raw = {name: float(r) for name, r, _ in rewards_cfg}
    weights = {name: float(w) for name, _, w in rewards_cfg}
    components = {name: float(r * w) for name, r, w in rewards_cfg}

    return sum(components.values()), components, raw, weights