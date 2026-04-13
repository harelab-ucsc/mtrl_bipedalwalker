from typing import Any, SupportsFloat

from Box2D import b2Vec2
import numpy as np
import math

from wrappers.bipedal_walker.walking_env import WalkReward


class WalkBackReward(WalkReward):
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
        joint_vel_l2 = np.mean([obs[5]**2, obs[7]**2, obs[10]**2, obs[12]**2])

        accel_x = hull_vel_x - self._prev_vel_x
        accel_y = hull_vel_y - self._prev_vel_y
        vel_jerk = (accel_x - self._prev_accel_x) ** 2 + (accel_y - self._prev_accel_y) ** 2

        # height above ground (interpolated terrain surface)
        ground_y = float(np.interp(hull_x, env.terrain_x, env.terrain_y))
        height_above_ground = env.hull.position.y - ground_y

        # penalize being close the ground
        TARGET_HEIGHT = 2 * (34 / 30.0) + 0.1  # 2 * LEG_H in world units
        height_err = TARGET_HEIGHT - height_above_ground
        body_height = max(height_err * abs(height_err), 0)  # signed squared error

        rewards_cfg: list[tuple[str, Any, float]] = [
            # coarse velocity tracking penalty
            ("vel_tracking", vel_tracking, -0.3),
            # fine velocity tracking reward
            ("vel_tracking_fine", vel_tracking_fine, 1.0),
            # penalize rotational velocity
            ("hull_ang_vel", hull_ang_vel, -0.1),
            # penalize deviation from upright
            ("hull_ang_l2", hull_ang_l2, -1.5),
            # penalize joint velocity
            ("joint_vel_l2", joint_vel_l2, -0.1),
            # body height reward. Once it reaches above the target, it becomes a reward. Otherwise it's a penalty.
            ("body_height", body_height, -0.1),
            # minimize velocity jerk
            ("vel_jerk", vel_jerk, -0.2),
            # penalize dying
            ("termination", termination, -150.0),
        ]

        components = {name: float(r * w) for name, r, w in rewards_cfg}
        return sum(components.values()), components

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict[str, Any]]:
        self._step_count = 0
        self._prev_vel_x = 0.0
        self._prev_vel_y = 0.0
        self._prev_accel_x = 0.0
        self._prev_accel_y = 0.0
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
