from typing import Any, SupportsFloat

from Box2D import b2Vec2
from gymnasium import Env
from gymnasium.core import ActType, ObsType
import numpy as np
import math

from pprint import pprint

from wrappers.bipedal_walker.walking_env import WalkReward


class WalkBackReward(WalkReward):
    def __init__(
        self,
        env: Env[ObsType, ActType],
        ep_time: int = 10,
        man_vel_ctrl: bool = False,
        vel_switching_freq: float = 10,
        vel_interp_speed: float = 1,
        vel_sample_range: tuple[float, float] = (-2.5, 2.5),
        vel_sample_zero: float = 0.2,
    ):
        super().__init__(
            env,
            ep_time,
            man_vel_ctrl,
            vel_switching_freq,
            vel_interp_speed,
            vel_sample_range,
            vel_sample_zero,
        )

        self._last_leg_contact = -1  # 0 -> left; 1 -> right; -1 -> unset
        self._last_obs_8 = 0.0
        self._last_obs_13 = 0.0
        self._steps_since_switch = 0

    def _compute_walk_rew(
        self, obs: np.ndarray, terminated: bool
    ) -> tuple[SupportsFloat, dict[str, float]]:
        """
        Observation layout (24 elements):
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
            [14:24]   lidar
        """

        env: Any = self.unwrapped
        hull_vel_x = env.hull.linearVelocity.x
        hull_vel_y = env.hull.linearVelocity.y
        hull_ang_vel = env.hull.angularVelocity
        hull_ang = env.hull.angle
        hull_x = env.hull.position.x

        # velocity tracking error
        vel_err = self._cmd_vel - hull_vel_x
        vel_tracking = vel_err**2
        # fine velocity tracking error
        vel_tracking_fine = 1 - np.tanh(40 * vel_tracking)
        
        # hull angle velocity
        hull_ang_vel = abs(hull_ang_vel) ** 2
        # hull angle deviation from 0
        hull_ang_l2 = hull_ang**2
        
        # termination
        termination = 1 if terminated else 0
        # minimize L2 joint_velocity
        joint_vel_l2 = np.mean([obs[5] ** 2, obs[7] ** 2, obs[10] ** 2, obs[12] ** 2])

        # print("leg 1 contact [before / after]:", self._last_obs_8, obs[8])
        # print("leg 2 contact [before / after]:", self._last_obs_13, obs[13])
        # print("last contact leg:", self._last_leg_contact)
        # print("step since switch:", self._steps_since_switch)
        # print()

        # leg alternating bonus
        # reward if the stepping goes left -> right -> left
        leg_alt_bonus = 0
        hopping_penalty = 0
        # initialize last leg contact if necessary
        if self._last_leg_contact == -1:  # ambiguous last
            if obs[8]:
                self._last_leg_contact = 0
            elif obs[13]:
                self._last_leg_contact = 1
        elif self._last_leg_contact == 0:  # last one is leg 1
            # leg 2 contact on rising edge = reward
            if obs[13] and not self._last_obs_13:
                # scale to stride length
                leg_alt_bonus = np.tanh(self._steps_since_switch / 30.0)
                self._last_leg_contact = 1  # switch to leg 2 now
                self._steps_since_switch = -1
            # leg 1 contact on rising edge again = penalty
            elif obs[8] and not self._last_obs_8:
                hopping_penalty = 1
                self._steps_since_switch = -1
        elif self._last_leg_contact == 1:  # last one is leg 2
            # leg 1 contact on rising edge = reward
            if obs[8] and not self._last_obs_8:
                # scale to stride length
                leg_alt_bonus = np.tanh(self._steps_since_switch / 30.0)
                self._last_leg_contact = 0  # switch to leg 1 now
                self._steps_since_switch = -1
            elif obs[13] and not self._last_obs_13:  # same leg again
                hopping_penalty = 1
                self._steps_since_switch = -1
        
        # only count when the last state is not ambiguous
        self._steps_since_switch += 0 if self._last_leg_contact == -1 else 1
            
        # print("leg 1 contact [before / after]:", self._last_obs_8, obs[8])
        # print("leg 2 contact [before / after]:", self._last_obs_13, obs[13])
        # print("last contact leg:", self._last_leg_contact + 1)
        # print("step since switch:", self._steps_since_switch)
        # print("hop penalty:", hopping_penalty * -0.3)
        # print("leg alt bonus:", leg_alt_bonus * 0.3)
        # print("\n ======================== \n")
        
        # update last contact states
        self._last_obs_8 = obs[8]
        self._last_obs_13 = obs[13]

        accel_x = hull_vel_x - self._prev_vel_x
        accel_y = hull_vel_y - self._prev_vel_y
        vel_jerk = (accel_x - self._prev_accel_x) ** 2 + (
            accel_y - self._prev_accel_y
        ) ** 2

        vel_y = hull_vel_y**2

        # height above ground (interpolated terrain surface)
        ground_y = float(np.interp(hull_x, env.terrain_x, env.terrain_y))
        height_above_ground = env.hull.position.y - ground_y

        # penalize being close the ground
        TARGET_HEIGHT = 2 * (34 / 30.0) + 0.1  # 2 * LEG_H in world units
        height_err = TARGET_HEIGHT - height_above_ground
        body_height = max(height_err * abs(height_err), 0)  # signed squared error

        rewards_cfg: list[tuple[str, Any, float]] = [
            # coarse velocity tracking penalty
            ("vel_tracking", vel_tracking, -0.1),
            # fine velocity tracking reward
            ("vel_tracking_fine", vel_tracking_fine, 1.0),
            # penalize rotational velocity
            ("hull_ang_vel", hull_ang_vel, -0.1),
            # penalize deviation from upright
            ("hull_ang_l2", hull_ang_l2, -2.0),
            # penalize joint velocity
            ("joint_vel_l2", joint_vel_l2, -0.05),
            # body height reward. Once it reaches above the target, it becomes a reward. Otherwise it's a penalty.
            ("body_height", body_height, -0.6),
            # penalize hull y velocity (don't bounce up and down)
            ("vel_y", vel_y, -0.1),
            # leg alternating bonus
            ("leg_alt_bonus", leg_alt_bonus, 1.0),
            # penalty for hopping
            ("hopping_penalty", hopping_penalty, -0.3),
            # minimize velocity jerk
            ("vel_jerk", vel_jerk, -0.1),
            # penalize dying
            ("termination", termination, -150.0),
        ]
        
        # for i in rewards_cfg:
        #     print(f"{i[0]}: {i[1] * i[2]}")
        # print()

        components = {name: float(r * w) for name, r, w in rewards_cfg}
        return sum(components.values()), components

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict[str, Any]]:
        self._step_count = 0
        self._prev_vel_x = 0.0
        self._prev_vel_y = 0.0
        self._prev_accel_x = 0.0
        self._prev_accel_y = 0.0
        self._last_leg_contact = -1
        self._steps_since_switch = 0
        self._last_obs_8 = 0.0
        self._last_obs_13 = 0.0

        obs, info = super(WalkReward, self).reset(seed=seed, options=options)

        # change command velocity
        if np.random.random() > self._vel_sample_zero:
            self._cmd_vel = np.random.uniform(*self._vel_sample_range)
        else:
            self._cmd_vel = 0.0

        self._cmd_vel_target = self._cmd_vel  # reset target
        self._cmd_vel_buf = [self._cmd_vel]  # reset buffer

        env: Any = self.unwrapped
        hull = env.hull
        legs = env.legs

        # all of these constants are defined here:
        # https://github.com/openai/gym/blob/bc212954b6713d5db303b3ead124de6cba66063e/gym/envs/box2d/bipedal_walker.py#L31
        VIEWPORT_H = 400
        SCALE = 30.0
        LEG_DOWN = -8 / SCALE
        LEG_H = 34 / SCALE

        # hip lim: (-0.8, 1.1)
        HIP_SAMPLE_LIM = (-0.3, 0.3)
        # knee lim: (-1.6, -0.1)
        KNEE_SAMPLE_LIM = (-0.3, -0.1)
        JOINT_VEL_SAMPLE_LIM = (-0.2, 0.2)
        # hull sampling
        HULL_Y_SAMPLE_LIM = (0.2, 0.3)
        HULL_X_SAMPLE_LIM = (40.0, 80.0)
        HULL_ROT_SAMPLE_LIM = (-0.2, 0.2)
        HULL_VEL_X_SAMPLE_LIM = (-0.2, 0.2)
        HULL_VEL_Y_SAMPLE_LIM = (0, 0)

        hull.position += b2Vec2(
            np.random.uniform(*HULL_X_SAMPLE_LIM), np.random.uniform(*HULL_Y_SAMPLE_LIM)
        )

        hull_x = env.hull.position.x
        ground_y = float(np.interp(hull_x, env.terrain_x, env.terrain_y))
        ground_y_rel = ground_y - VIEWPORT_H / SCALE / 4
        # move hull to above ground
        hull.position.y += ground_y_rel

        hull.linearVelocity += b2Vec2(
            np.random.uniform(*HULL_VEL_X_SAMPLE_LIM),
            np.random.uniform(*HULL_VEL_Y_SAMPLE_LIM),
        )
        hull.angle += np.random.uniform(*HULL_ROT_SAMPLE_LIM)

        # get hull position and angle to calculate joint pos
        hull_a = hull.angle
        hull_x, hull_y = hull.position

        # reference angles baked at joint creation: bodyB.angle - bodyA.angle
        # https://github.com/openai/gym/blob/bc212954b6713d5db303b3ead124de6cba66063e/gym/envs/box2d/bipedal_walker.py#L459
        # pair 0 (i=-1): hip_ref = -0.05, knee_ref = 0
        # pair 1 (i=+1): hip_ref = +0.05, knee_ref = 0
        hip_refs = [-0.05, 0.05]

        for pair in range(2):
            upper = legs[pair * 2]
            lower = legs[pair * 2 + 1]

            hip_angle = np.random.uniform(*HIP_SAMPLE_LIM)
            knee_angle = np.random.uniform(*KNEE_SAMPLE_LIM)

            # world-space body angles via joint.angle = bB.angle - bA.angle - refAngle
            upper_a = hull_a + hip_refs[pair] + hip_angle
            lower_a = upper_a + 0.0 + knee_angle  # knee ref = 0

            # hip anchor world pos (hull local anchor = (0, LEG_DOWN))
            hip_wx = hull_x - LEG_DOWN * math.sin(hull_a)
            hip_wy = hull_y + LEG_DOWN * math.cos(hull_a)

            # upper leg center (its local anchor to hip = (0, LEG_H/2))
            upper_x = hip_wx + (LEG_H / 2) * math.sin(upper_a)
            upper_y = hip_wy - (LEG_H / 2) * math.cos(upper_a)

            # knee anchor world pos (upper leg local anchor = (0, -LEG_H/2))
            knee_wx = upper_x + (LEG_H / 2) * math.sin(upper_a)
            knee_wy = upper_y - (LEG_H / 2) * math.cos(upper_a)

            # lower leg center (its local anchor to knee = (0, LEG_H/2))
            lower_x = knee_wx + (LEG_H / 2) * math.sin(lower_a)
            lower_y = knee_wy - (LEG_H / 2) * math.cos(lower_a)

            for body, bx, by, ba in [
                (upper, upper_x, upper_y, upper_a),
                (lower, lower_x, lower_y, lower_a),
            ]:
                body.position = (bx, by)
                body.angle = ba
                body.linearVelocity = (0, 0)
                body.angularVelocity = np.random.uniform(*JOINT_VEL_SAMPLE_LIM)
                body.awake = True

        # apply the changes
        obs = env.step(np.array([0, 0, 0, 0]))[0]

        # append command velocity to observations
        obs = np.append(obs, np.float64(self._cmd_vel))

        return obs, info
