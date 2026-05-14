from typing import Any, SupportsFloat

from gymnasium import Env, Wrapper
from gymnasium.core import ActType, ObsType
from Box2D import b2Vec2

import numpy as np
import math

from gymnasium import spaces
import pygame


class BodyTiltEnv(Wrapper):
    def __init__(
        self,
        env: Env[ObsType, ActType],
        ep_time: int = 10,
        ang_switching_freq: float = 10,
        ang_interp_speed: float = 1,
        ang_sample_range: tuple[float, float] = (-0.75, 0.75),
        ang_sample_zero: float = 0.15,
        hull_x_range: tuple[float, float] = (20.0, 60.0),
    ):
        super().__init__(env)

        # specified here: https://github.com/openai/gym/blob/bc212954b6713d5db303b3ead124de6cba66063e/gym/envs/box2d/bipedal_walker.py#L30
        FPS = 50

        self._max_steps: int = ep_time * FPS
        self._step_count: int = 0

        self._ang_switch_steps: int = np.floor(ang_switching_freq * FPS)
        self._ang_sample_range: tuple[float, float] = ang_sample_range
        self._ang_sample_zero: float = ang_sample_zero
        self._hull_x_range: tuple[float, float] = hull_x_range

        self._difficulty: float = 0.0

        # initialize tilt commands
        self._cmd_tilt: float = 0
        self._cmd_tilt_target: float = 0
        self._cmd_tilt_buf: list[float] = [0]  # smooth out transitions
        self._max_cmd_buf_size: int = np.floor(
            ang_interp_speed * FPS
        )  # smooth out transitions

        base = self.env.observation_space
        self.observation_space = spaces.Box(
            low=np.append(base.low, -np.inf),  # type: ignore
            high=np.append(base.high, np.inf),  # type: ignore
            dtype=np.float64,
        )

    def _compute_body_tilt_rew(
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
        """

        env: Any = self.unwrapped
        hull_vel_x = env.hull.linearVelocity.x
        hull_vel_y = env.hull.linearVelocity.y
        hull_ang_vel = env.hull.angularVelocity
        hull_ang = env.hull.angle
        hull_x = env.hull.position.x

        # hull angle tracking error
        hull_ang_err = self._cmd_tilt - hull_ang
        hull_ang_tracking = hull_ang_err**2
        # fine hull angle tracking error
        hull_ang_tracking_fine = 1 - np.tanh(40 * hull_ang_tracking)
        # hull angle velocity
        hull_ang_vel = abs(hull_ang_vel) ** 2
        # termination
        termination = 1 if terminated else 0
        # minimize L2 joint_velocity
        joint_vel_l2 = np.mean([obs[5] ** 2, obs[7] ** 2, obs[10] ** 2, obs[12] ** 2])

        # height above ground (interpolated terrain surface)
        ground_y = float(np.interp(hull_x, env.terrain_x, env.terrain_y))
        height_above_ground = env.hull.position.y - ground_y

        # penalize being close to the ground
        TARGET_HEIGHT = 2.25 * (34 / 30.0)  # 2 * LEG_H — stand tall
        height_err = TARGET_HEIGHT - height_above_ground
        body_height = max(height_err * abs(height_err), 0)  # signed squared error

        # penalize horizontal drift velocity
        hull_vel_x_l2 = hull_vel_x**2
        # penalize vertical instability
        hull_vel_y_l2 = hull_vel_y**2

        rewards_cfg: list[tuple[str, Any, float]] = [
            # coarse tilt tracking penalty
            ("hull_ang_tracking", hull_ang_tracking, -0.3),
            # fine tilt tracking reward
            ("hull_ang_tracking_fine", hull_ang_tracking_fine, 1.0),
            # penalize rotational velocity
            ("hull_ang_vel", hull_ang_vel, -0.1),
            # penalize joint velocity
            ("joint_vel_l2", joint_vel_l2, -0.005),
            # body height reward. Once it reaches above the target, it becomes a reward. Otherwise it's a penalty.
            ("body_height", body_height, -0.4),
            # penalize horizontal drift velocity
            ("hull_vel_x_l2", hull_vel_x_l2, -0.1),
            # penalize vertical instability (bouncing, oscillation)
            ("hull_vel_y_l2", hull_vel_y_l2, -0.07),
            # penalize dying
            ("termination", termination, -150.0),
        ]

        raw = {name: float(r) for name, r, _ in rewards_cfg}
        weights = {name: float(w) for name, _, w in rewards_cfg}
        components = {name: float(r * w) for name, r, w in rewards_cfg}

        return sum(components.values()), components, raw, weights

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # step environment
        obs, rew, term, trunc, info = super().step(action)

        # detect truncation
        self._step_count += 1
        trunc = trunc or self._step_count >= self._max_steps
        rew, info["reward_terms"], info["reward_raw"], info["reward_weights"] = (
            self._compute_body_tilt_rew(obs, term)
        )

        # change command velocity if specified
        if (
            self._ang_switch_steps < self._max_steps
            and self._step_count % self._ang_switch_steps == 0
        ):
            if np.random.random() > self._ang_sample_zero:
                self._cmd_tilt_target = np.random.uniform(*self._ang_sample_range)
            else:
                self._cmd_tilt_target = 0.0

        # push the target command to the buf
        self._cmd_tilt_buf.append(self._cmd_tilt_target)
        if len(self._cmd_tilt_buf) > self._max_cmd_buf_size:
            # trim front
            self._cmd_tilt_buf.pop(0)

        # update cmd tilt
        self._cmd_tilt = float(np.mean(self._cmd_tilt_buf))

        # append command tilt to observations
        obs = np.append(obs, np.float64(self._cmd_tilt))

        return obs, rew, term, trunc, info

    def render(self):
        result = super().render()  # gets rgb_array frame with base rendering done

        env: Any = self.unwrapped
        if not hasattr(env, "surf") or env.surf is None:
            return result

        self._draw_velocity_arrows(env)

        # re-grab the frame after drawing on surf
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(env.surf)), axes=(1, 0, 2)
        )[:, -600:]
        
    def _draw_velocity_arrows(self, env):
        SCALE = 30.0
        VIEWPORT_H = 400
        HULL_FRONT_OFFSET = 25
        BLUE_ARROW_LEN = 45
        GREEN_ARROW_LEN = 30
        HEAD_LEN = 10
        HEAD_WIDTH = 5

        unwrapped: Any = self.unwrapped
        hull_ang = unwrapped.hull.angle
        hull_x, hull_y = unwrapped.hull.position
        cx = hull_x * SCALE
        cy = VIEWPORT_H - hull_y * SCALE

        def screen_dir(angle):
            return math.cos(angle), -math.sin(angle)

        def draw_segment(start, end, color):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-6:
                return
            ux, uy = dx / length, dy / length
            shaft_end = (end[0] - ux * HEAD_LEN, end[1] - uy * HEAD_LEN)
            pygame.draw.line(
                env.surf, color,
                (int(start[0]), int(start[1])),
                (int(shaft_end[0]), int(shaft_end[1])), 2,
            )
            px, py = -uy, ux
            tip = (int(end[0]), int(end[1]))
            p2 = (int(end[0] - ux * HEAD_LEN + px * HEAD_WIDTH), int(end[1] - uy * HEAD_LEN + py * HEAD_WIDTH))
            p3 = (int(end[0] - ux * HEAD_LEN - px * HEAD_WIDTH), int(end[1] - uy * HEAD_LEN - py * HEAD_WIDTH))
            pygame.draw.polygon(env.surf, color, [tip, p2, p3])

        # shared anchor: slightly in front of hull center along actual hull angle
        bx, by = screen_dir(hull_ang)
        anchor = (cx + HULL_FRONT_OFFSET * bx, cy + HULL_FRONT_OFFSET * by)
        blue_end = (anchor[0] + BLUE_ARROW_LEN * bx, anchor[1] + BLUE_ARROW_LEN * by)

        # green pivots from the same anchor, pointing in command direction (longer so it's visible when aligned)
        gx, gy = screen_dir(self._cmd_tilt)
        green_end = (anchor[0] + (BLUE_ARROW_LEN + GREEN_ARROW_LEN) * gx, anchor[1] + (BLUE_ARROW_LEN + GREEN_ARROW_LEN) * gy)

        draw_segment(anchor, blue_end, (71, 126, 255))   # blue = actual hull angle
        draw_segment(anchor, green_end, (9, 176, 12))    # green = command tilt

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict[str, Any]]:
        self._step_count = 0
        obs, info = super().reset(seed=seed, options=options)

        # change command velocity
        if np.random.random() > self._ang_sample_zero:
            self._cmd_tilt = np.random.uniform(*self._ang_sample_range)
        else:
            self._cmd_tilt = 0.0

        self._cmd_tilt_target = self._cmd_tilt  # reset target
        self._cmd_tilt_buf = [self._cmd_tilt]  # reset buffer

        env: Any = self.unwrapped
        hull = env.hull
        legs = env.legs

        # all of these constants are defined here:
        # https://github.com/openai/gym/blob/bc212954b6713d5db303b3ead124de6cba66063e/gym/envs/box2d/bipedal_walker.py#L31
        VIEWPORT_H = 400
        SCALE = 30.0
        LEG_DOWN = -8 / SCALE
        LEG_H = 34 / SCALE

        d = self._difficulty

        def lerp(a: float, b: float) -> float:
            return a + (b - a) * d

        # hip lim: (-0.8, 1.1) — easy→hard
        HIP_SAMPLE_LIM = (lerp(-0.3, -0.7), lerp(0.3, 0.9))
        # knee lim: (-1.6, -0.1) — easy→hard
        KNEE_SAMPLE_LIM = (lerp(-0.3, -1.4), -0.1)
        JOINT_VEL_SAMPLE_LIM = (lerp(-0.2, -1.5), lerp(0.2, 1.5))
        # hull sampling
        HULL_Y_SAMPLE_LIM = (0.2, 0.3)
        HULL_ROT_SAMPLE_LIM = (lerp(-0.2, -0.65), lerp(0.2, 0.65))
        HULL_VEL_X_SAMPLE_LIM = (lerp(-0.2, -0.5), lerp(0.2, 0.5))
        HULL_VEL_Y_SAMPLE_LIM = (lerp(0.0, -0.3), lerp(0.0, 0.3))

        hull.position += b2Vec2(
            np.random.uniform(*self._hull_x_range),
            np.random.uniform(*HULL_Y_SAMPLE_LIM),
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
        obs = np.append(obs, np.float64(self._cmd_tilt))

        return obs, info
