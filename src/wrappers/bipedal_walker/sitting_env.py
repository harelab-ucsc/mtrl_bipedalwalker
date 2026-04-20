from typing import Any, SupportsFloat

from gymnasium import Env, Wrapper
from gymnasium.core import ActType, ObsType
from Box2D import b2Vec2

import numpy as np
import math

from mdp.bipedal_walker.rewards import body_lin_vel_l2, leg_contact


class SitReward(Wrapper):
    """
    Wrapper that replaces BipedalWalker's default reward with a sitting reward.

    The target pose is a low crouch: hull near the ground, upright, both feet
    planted, knees fully folded, no motion. Observation space is unchanged
    (24 elements) — there is no velocity command.

    Args:
        env: The BipedalWalker environment to wrap.
        ep_time: Maximum episode duration in seconds. Default: 10.
        disturbance_freq: Frequency of external force disturbance in seconds.
            -1 disables. Default: -1.
        disturbance_force: Range of forces as ((x_min,x_max),(y_min,y_max)).
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        ep_time: int = 10,
        disturbance_freq: int = -1,
        disturbance_force: tuple[tuple[float, float], tuple[float, float]] = (
            (0, 0),
            (0, 0),
        ),
    ):
        super().__init__(env)

        # specified here: https://github.com/openai/gym/blob/bc212954b6713d5db303b3ead124de6cba66063e/gym/envs/box2d/bipedal_walker.py#L30
        FPS = 50

        self._max_steps: int = ep_time * FPS
        self._step_count: int = 0
        self._disturb_freq: int = disturbance_freq * FPS
        self._disturb_force: tuple[tuple[float, float], tuple[float, float]] = (
            disturbance_force
        )
        self._last_disturbance: tuple[float, float] | None = None
        self._disturbance_display_frames: int = 0
        self._prev_action: np.ndarray = np.zeros(4, dtype=np.float32)

    def _compute_sit_rew(
        self, obs: np.ndarray, action: np.ndarray, terminated: bool
    ) -> tuple[SupportsFloat, dict[str, float], dict[str, float], dict[str, float]]:
        """
        Observation layout (24 elements):
            [0]       hull_ang
            [1]       hull_ang_vel
            [2]       vel_x             (normalized, do not use)
            [3]       vel_y             (normalized, do not use)
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

        Sitting reward terms:
            - Minimize linear velocity (use the real b2 hull vel, not the
              normalized obs values).
            - Minimize rotational velocity.
            - Both feet contacting ground.
            - Hull upright (hull_ang L2, normalized by pi/2).
            - Hull height close to the low sit target.
            - Knees folded (obs knee positions ~ -1.0 in joint space).
            - Minimize joint velocity.
            - Penalize termination.
        """

        env: Any = self.unwrapped

        # real linear velocity (the obs versions are normalized)
        hull_vel_x = env.hull.linearVelocity.x
        hull_vel_y = env.hull.linearVelocity.y
        vel_l2 = body_lin_vel_l2((np.float32(hull_vel_x), np.float32(hull_vel_y)))

        hull_ang_vel = abs(obs[1]) ** 2
        leg_contacts = leg_contact((obs[8], obs[13]))  # 0..1, want both
        hull_ang_l2 = obs[0] ** 2
        termination = 1 if terminated else 0
        joint_vel = abs(np.mean([obs[5], obs[7], obs[10], obs[12]]))

        # height above ground (interpolated terrain surface)
        hull_x = env.hull.position.x
        ground_y = float(np.interp(hull_x, env.terrain_x, env.terrain_y))
        height_above_ground = env.hull.position.y - ground_y

        # target: the steady-state the policy naturally converges to with
        # bent knees (~1.2*LEG_H measured). Lower targets (e.g. 1.0*LEG_H)
        # are below the geometric minimum and create a permanent pull toward
        # an unreachable pose, causing instability.
        LEG_H = 34 / 30.0
        SIT_TARGET_HEIGHT = 1.2 * LEG_H
        body_height_err = (SIT_TARGET_HEIGHT - height_above_ground) ** 2

        # knee fold: reward knee joint positions close to the folded end
        # of their range. Joint range is (-1.6, -0.1); -1.2 is a deep fold.
        KNEE_FOLD_TARGET = -1.2
        knee_fold_err = (
            (obs[6] - KNEE_FOLD_TARGET) ** 2 + (obs[11] - KNEE_FOLD_TARGET) ** 2
        )

        # normalize some rewards
        hull_ang_l2 /= np.pi / 2  # ±90deg -> ±1
        hull_ang_vel /= 1e-2  # empirically measured (matches StandReward)

        # action-magnitude and action-smoothness penalties — without these
        # PPO happily finds a degenerate hold pose where it pumps opposing
        # torques every step, causing visible jitter even though the joints
        # barely move.
        action_l2 = float(np.mean(np.asarray(action, dtype=np.float32) ** 2))
        action_delta = float(
            np.mean(
                (np.asarray(action, dtype=np.float32) - self._prev_action) ** 2
            )
        )

        rewards_cfg: list[tuple[str, Any, float]] = [
            # constant per-step bonus so staying alive is strictly positive.
            # Without this, every reward term is a penalty and the agent
            # learns "die fast" to cut losses.
            ("alive", 1.0, 1.0),
            ("vel_l2", vel_l2, -0.3),
            ("hull_ang_vel", hull_ang_vel, -0.1),
            ("leg_contacts", leg_contacts, 0.3),
            ("hull_ang_l2", hull_ang_l2, -1.0),
            ("body_height_err", body_height_err, -0.8),
            ("knee_fold_err", knee_fold_err, -0.3),
            ("joint_vel", joint_vel, -0.3),
            ("action_l2", action_l2, -0.15),
            ("action_delta", action_delta, -0.5),
            ("termination", termination, -300.0),
        ]

        raw = {name: float(r) for name, r, w in rewards_cfg}
        weights = {name: float(w) for name, r, w in rewards_cfg}
        components = {name: float(r * w) for name, r, w in rewards_cfg}
        return sum(components.values()), components, raw, weights

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, rew, term, trunc, info = super().step(action)

        self._step_count += 1
        trunc = trunc or self._step_count >= self._max_steps
        action_arr = np.asarray(action, dtype=np.float32)
        rew, info["reward_terms"], info["reward_raw"], info["reward_weights"] = self._compute_sit_rew(obs, action_arr, term)
        self._prev_action = action_arr

        # apply periodic disturbances if enabled
        if self._disturbance_display_frames > 0:
            self._disturbance_display_frames -= 1

        if (
            self._disturb_freq > 0
            and self._step_count % self._disturb_freq == 0
            and self._step_count > 0
        ):
            _env: Any = self.unwrapped
            hull = _env.hull
            force = b2Vec2(
                np.random.uniform(*self._disturb_force[0]),
                np.random.uniform(*self._disturb_force[1]),
            )
            hull.linearVelocity += force
            self._last_disturbance = (force.x, force.y)
            self._disturbance_display_frames = 10

        return obs, rew, term, trunc, info

    def render(self):
        result = super().render()

        if self._last_disturbance is None or self._disturbance_display_frames <= 0:
            return result

        import pygame

        env: Any = self.unwrapped
        if not hasattr(env, "surf") or env.surf is None:
            return result

        self._draw_disturbance_arrow(pygame, env)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(env.surf)), axes=(1, 0, 2)
        )[:, -600:]

    def _draw_disturbance_arrow(self, pygame, env):
        SCALE = 30.0
        VIEWPORT_H = 400

        hull_x, hull_y = env.hull.position
        sx = hull_x * SCALE
        sy = VIEWPORT_H - hull_y * SCALE

        assert self._last_disturbance is not None
        fx, fy = self._last_disturbance
        mag = math.sqrt(fx * fx + fy * fy)
        if mag < 1e-6:
            return

        color = (255, 0, 0)

        ARROW_SCALE = 10
        dsx, dsy = fx, -fy
        ex = sx + dsx * ARROW_SCALE
        ey = sy + dsy * ARROW_SCALE

        pygame.draw.line(env.surf, color, (int(sx), int(sy)), (int(ex), int(ey)), 2)

        head_back, head_width = 5, 3
        nx, ny = fx / mag, fy / mag
        psx, psy = ny, -nx
        p1 = (int(ex), int(ey))
        p2 = (
            int(ex - nx * head_back + psx * head_width),
            int(ey + ny * head_back + psy * head_width),
        )
        p3 = (
            int(ex - nx * head_back - psx * head_width),
            int(ey + ny * head_back - psy * head_width),
        )
        pygame.draw.polygon(env.surf, color, [p1, p2, p3])

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict[str, Any]]:
        self._step_count = 0
        self._last_disturbance = None
        self._disturbance_display_frames = 0
        self._prev_action = np.zeros(4, dtype=np.float32)
        obs, info = super().reset(seed=seed, options=options)

        env: Any = self.unwrapped
        hull = env.hull
        legs = env.legs

        # constants from bipedal_walker.py
        SCALE = 30.0
        LEG_DOWN = -8 / SCALE
        LEG_H = 34 / SCALE

        HIP_SAMPLE_LIM = (-0.8, 1.1)
        KNEE_SAMPLE_LIM = (-1.6, -0.1)
        HULL_Y_SAMPLE_LIM = (0.35, 0.55)
        HULL_ROT_SAMPLE_LIM = (-0.2, 0.2)
        HULL_VEL_X_SAMPLE_LIM = (-3, 3)
        HULL_VEL_Y_SAMPLE_LIM = (-0.6, 0.6)

        JOINT_VEL_SAMPLE_LIM = (-0.2, 0.2)
        HULL_X_SAMPLE_LIM = (3.0, 8.0)

        hull.position += b2Vec2(
            np.random.uniform(*HULL_X_SAMPLE_LIM),
            np.random.uniform(*HULL_Y_SAMPLE_LIM),
        )
        hull.linearVelocity += b2Vec2(
            np.random.uniform(*HULL_VEL_X_SAMPLE_LIM),
            np.random.uniform(*HULL_VEL_Y_SAMPLE_LIM),
        )
        hull.angle += np.random.uniform(*HULL_ROT_SAMPLE_LIM)

        hull_a = hull.angle
        hull_x, hull_y = hull.position

        hip_refs = [-0.05, 0.05]

        for pair in range(2):
            upper = legs[pair * 2]
            lower = legs[pair * 2 + 1]

            hip_angle = np.random.uniform(*HIP_SAMPLE_LIM)
            knee_angle = np.random.uniform(*KNEE_SAMPLE_LIM)

            upper_a = hull_a + hip_refs[pair] + hip_angle
            lower_a = upper_a + 0.0 + knee_angle

            hip_wx = hull_x - LEG_DOWN * math.sin(hull_a)
            hip_wy = hull_y + LEG_DOWN * math.cos(hull_a)

            upper_x = hip_wx + (LEG_H / 2) * math.sin(upper_a)
            upper_y = hip_wy - (LEG_H / 2) * math.cos(upper_a)

            knee_wx = upper_x + (LEG_H / 2) * math.sin(upper_a)
            knee_wy = upper_y - (LEG_H / 2) * math.cos(upper_a)

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

        # apply the teleport
        obs = env.step(np.array([0, 0, 0, 0]))[0]
        return obs, info
