from typing import Any, SupportsFloat

from gymnasium import Env, Wrapper
from gymnasium.core import ActType, ObsType
from Box2D import b2Vec2

import numpy as np
import math

from gymnasium import spaces


class HopReward(Wrapper):
    def __init__(
        self,
        env: Env[ObsType, ActType],
        ep_time: int = 10,
        man_vel_ctrl: bool = False,
        vel_switching_freq: int = 10,
        vel_sample_range: tuple[float, float] = (-2.5, 2.5),
        vel_sample_zero: float = 0.2,
    ):
        """
        Wrapper that replaces BipedalWalker's default reward with a hopping reward.

        Args:
        env: The BipedalWalker environment to wrap.

        ep_time: Maximum episode duration in seconds. Default: 15.

        man_vel_ctrl: Manual velocity control. Keep False for training. Default: False.

        vel_switching_freq: Frequency in seconds in which velocity command is switched. This only has an effect when man_vel_ctrl is False. Default: 10.

        vel_sample_range: The range from which to sample command velocity from. This only has an effect when man_vel_ctrl is False. Default: [-1, 1].

        vel_sample_zero: Probability in sampling a 0 velocity command.  This only has an effect when man_vel_ctrl is False. Default: 0.2.
        """
        super().__init__(env)

        # specified here: https://github.com/openai/gym/blob/bc212954b6713d5db303b3ead124de6cba66063e/gym/envs/box2d/bipedal_walker.py#L30
        FPS = 50

        self._max_steps: int = ep_time * FPS
        self._step_count: int = 0

        self._man_vel_ctrl: bool = man_vel_ctrl
        self._vel_switch_steps: int = vel_switching_freq * FPS
        self._vel_sample_range: tuple[float, float] = vel_sample_range
        self._vel_sample_zero: float = vel_sample_zero

        # initialize velocity commands
        self._cmd_vel: float = 0
        # TODO: implement a cmd_vel smoothing buffer if necessary (to smoothly switch btwn one command vel to another)
        
        base = self.env.observation_space
        self.observation_space = spaces.Box(
            low=np.append(base.low, -np.inf), # type: ignore
            high=np.append(base.high, np.inf), # type: ignore
            dtype=np.float64
        )

    def _compute_hop_rew(
        self, obs: np.ndarray, terminated: bool
    ) -> tuple[SupportsFloat, dict[str, float]]:
        """
        Observation layout (24 elements):
            [0]       hull_ang
            [1]       hull_ang_vel
            [2]       vel_x
            [3]       vel_y
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

        # velocity tracking error
        vel_err = self._cmd_vel - obs[2]
        vel_tracking = vel_err ** 2
        # fine velocity tracking error
        vel_tracking_fine = 1 - np.tanh(30 * vel_tracking)
        # hull angle velocity
        hull_ang_vel = abs(obs[1]) ** 2
        # leg lift
        leg_1_contact = 1 if obs[8] == 0 and obs[13] == 0 else 0  # encourage leg 1 air time
        leg_2_contact = 1 if obs[13] == 1 else 0  # penalize any leg 2 contact
        # hull angle deviation from 0
        hull_ang_l2 = obs[0] ** 2
        # termination
        termination = 1 if terminated else 0
        # minimize L2 joint_velocity
        joint_vel_l2 = (np.mean([obs[5], obs[7], obs[10], obs[12]])) ** 2

        rewards_cfg: list[tuple[str, Any, float]] = [
            # coarse velocity tracking penalty
            ("vel_tracking", vel_tracking, -0.3),
            # fine velocity tracking reward
            ("vel_tracking_fine", vel_tracking_fine, 1.0),
            # penalize rotational velocity
            ("hull_ang_vel", hull_ang_vel, -0.1),
            # reward strict single-leg contact
            ("leg_1_contact", leg_1_contact, 0.2),
            ("leg_2_contact", leg_2_contact, -0.2),
            # penalize deviation from upright
            ("hull_ang_l2", hull_ang_l2, -0.5),
            # penalize joint velocity
            ("joint_vel_l2", joint_vel_l2, -0.05),
            # penalize dying
            ("termination", termination, -20.0),
        ]

        components = {name: float(r * w) for name, r, w in rewards_cfg}
        return sum(components.values()), components

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # step environment
        obs, rew, term, trunc, info = super().step(action)

        # detect truncation
        self._step_count += 1
        trunc = trunc or self._step_count >= self._max_steps
        rew, info["reward_terms"] = self._compute_hop_rew(obs, term)

        # change command velocity if specified
        if (
            self._vel_switch_steps < self._max_steps
            and self._step_count % self._vel_switch_steps == 0
        ):
            if np.random.random() > self._vel_sample_zero:
                self._cmd_vel = np.random.uniform(*self._vel_sample_range)
            else:
                self._cmd_vel = 0.0
                
        # append command velocity to observations
        obs = np.append(obs, np.float64(self._cmd_vel))

        return obs, rew, term, trunc, info

    def render(self):
        result = super().render()  # gets rgb_array frame with base rendering done

        import pygame

        env: Any = self.unwrapped
        if not hasattr(env, "surf") or env.surf is None:
            return result

        self._draw_velocity_arrows(pygame, env)

        # re-grab the frame after drawing on surf
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(env.surf)), axes=(1, 0, 2)
        )[:, -600:]

    def _draw_velocity_arrows(self, pygame, env):
        # TODO: pass or cache current obs so real_vel_x is accessible here
        unwrapped: Any = self.unwrapped
        real_vel_x: float = unwrapped.hull.linearVelocity.x


        # green = command
        self._draw_velocity_arrow(pygame, env, self._cmd_vel, color=(9, 176, 12), y_offset=-10)
        # blue = real
        self._draw_velocity_arrow(pygame, env, real_vel_x, color=(71, 126, 255))

    def _draw_velocity_arrow(
        self,
        pygame,
        env,
        vel_x: float,
        color: tuple,
        y_offset: int = 0
    ):
        SCALE = 30.0
        VIEWPORT_H = 400
        ARROW_SCALE = 2
        
        HEAD_LEN = 10
        HEAD_WIDTH = 5

        hull_x, hull_y = env.hull.position
        sx = hull_x * SCALE
        sy = VIEWPORT_H - hull_y * SCALE - 40 + y_offset

        if abs(vel_x) < 1e-6:
            return

        sign = math.copysign(1.0, vel_x)   # direction
        mag = abs(vel_x) * ARROW_SCALE * 10 # pixel length of shaft

        ex = sx + sign * mag
        ey = sy

        # shorten shaft so it ends at base of head, not tip
        shaft_ex = ex - sign * HEAD_LEN
        pygame.draw.line(env.surf, color, (int(sx), int(sy)), (int(shaft_ex), int(ey)), 2)

        # constant-size arrowhead: tip at (ex, ey), base perpendicular in screen y
        p1 = (int(ex), int(ey))
        p2 = (int(ex - sign * HEAD_LEN), int(ey - HEAD_WIDTH))
        p3 = (int(ex - sign * HEAD_LEN), int(ey + HEAD_WIDTH))

        pygame.draw.polygon(env.surf, color, [p1, p2, p3])

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict[str, Any]]:
        self._step_count = 0
        obs, info = super().reset(seed=seed, options=options)

        # change command velocity
        if np.random.random() > self._vel_sample_zero:
            self._cmd_vel = np.random.uniform(*self._vel_sample_range)
        else:
            self._cmd_vel = 0.0

        env: Any = self.unwrapped
        hull = env.hull
        legs = env.legs

        # all of these constants are defined here:
        # https://github.com/openai/gym/blob/bc212954b6713d5db303b3ead124de6cba66063e/gym/envs/box2d/bipedal_walker.py#L31
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
        HULL_X_SAMPLE_LIM = (3.0, 8.0)
        HULL_ROT_SAMPLE_LIM = (-0.1, 0.1)
        HULL_VEL_X_SAMPLE_LIM = (-0.2, 0.2)
        HULL_VEL_Y_SAMPLE_LIM = (0, 0)

        hull.position += b2Vec2(
            np.random.uniform(*HULL_X_SAMPLE_LIM), np.random.uniform(*HULL_Y_SAMPLE_LIM)
        )
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
