from typing import Any, SupportsFloat

from gymnasium import Env
from gymnasium.core import ActType, ObsType
import numpy as np

from wrappers.bipedal_walker.walk_env import WalkEnv


class WalkFTEnv(WalkEnv):
    def __init__(
        self,
        env: Env[ObsType, ActType],
        ep_time: int = 10,
        man_vel_ctrl: bool = False,
        vel_switching_freq: float = 10,
        vel_interp_speed: float = 1,
        vel_sample_range: tuple[float, float] = (-2.5, 2.5),
        vel_sample_zero: float = 0.2,
        hull_x_range: tuple[float, float] = (0.0, 40.0),
    ):
        """
        Fine-tuning walk landscape. Extends WalkEnv with leg alternation bonus/penalty
        and vertical velocity penalty for a smoother, more bipedal gait.

        Use hull_x_range=(40.0, 80.0) for backward-direction tasks.

        Args:
        env: The BipedalWalker environment to wrap.
        ep_time: Maximum episode duration in seconds. Default: 10.
        man_vel_ctrl: Manual velocity control. Keep False for training. Default: False.
        vel_switching_freq: Frequency in seconds in which velocity command is switched. Default: 10.
        vel_interp_speed: Amount of time in seconds to interpolate between two velocity commands. Default: 1.
        vel_sample_range: The range from which to sample command velocity. Default: (-2.5, 2.5).
        vel_sample_zero: Probability of sampling a 0 velocity command. Default: 0.2.
        hull_x_range: X spawn range for domain randomization. Default: (0.0, 40.0).
        """
        super().__init__(
            env,
            ep_time,
            man_vel_ctrl,
            vel_switching_freq,
            vel_interp_speed,
            vel_sample_range,
            vel_sample_zero,
            hull_x_range,
        )

        self._last_leg_contact = -1  # 0 -> left; 1 -> right; -1 -> unset
        self._last_obs_8 = 0.0
        self._last_obs_13 = 0.0
        self._steps_since_switch = 0

    def _compute_walk_rew(
        self, obs: np.ndarray, terminated: bool
    ) -> tuple[SupportsFloat, dict[str, float], dict[str, float], dict[str, float]]:
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

        # leg alternating bonus: reward if the stepping goes left -> right -> left
        leg_alt_bonus = 0
        hopping_penalty = 0
        if self._last_leg_contact == -1:  # ambiguous last
            if obs[8]:
                self._last_leg_contact = 0
            elif obs[13]:
                self._last_leg_contact = 1
        elif self._last_leg_contact == 0:  # last one is leg 1
            # leg 2 contact on rising edge = reward
            if obs[13] and not self._last_obs_13:
                leg_alt_bonus = np.tanh(self._steps_since_switch / 30.0)
                self._last_leg_contact = 1
                self._steps_since_switch = -1
            # leg 1 contact on rising edge again = penalty
            elif obs[8] and not self._last_obs_8:
                hopping_penalty = 1
                self._steps_since_switch = -1
        elif self._last_leg_contact == 1:  # last one is leg 2
            # leg 1 contact on rising edge = reward
            if obs[8] and not self._last_obs_8:
                leg_alt_bonus = np.tanh(self._steps_since_switch / 30.0)
                self._last_leg_contact = 0
                self._steps_since_switch = -1
            elif obs[13] and not self._last_obs_13:  # same leg again
                hopping_penalty = 1
                self._steps_since_switch = -1

        # only count when the last state is not ambiguous
        self._steps_since_switch += 0 if self._last_leg_contact == -1 else 1

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

        # penalize being close to the ground
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
            # penalty for hopping (same leg landing twice in a row)
            ("hopping_penalty", hopping_penalty, -0.3),
            # minimize velocity jerk
            ("vel_jerk", vel_jerk, -0.1),
            # penalize dying
            ("termination", termination, -150.0),
        ]

        raw = {name: float(r) for name, r, w in rewards_cfg}
        weights = {name: float(w) for name, r, w in rewards_cfg}
        components = {name: float(r * w) for name, r, w in rewards_cfg}
        return sum(components.values()), components, raw, weights

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict[str, Any]]:
        self._last_leg_contact = -1
        self._steps_since_switch = 0
        self._last_obs_8 = 0.0
        self._last_obs_13 = 0.0
        return super().reset(seed=seed, options=options)
