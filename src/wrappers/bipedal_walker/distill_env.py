from typing import Any, SupportsFloat

from gymnasium import Env
import pygame
from wrappers.bipedal_walker.proprio_wrapper import ProprioObsWrapper
from gymnasium.core import ActType, ObsType
from Box2D import b2Vec2

import numpy as np
import math


class DistillEnv(ProprioObsWrapper):
    def __init__(
        self,
        env: Env[ObsType, ActType],
        ep_time: int = 10,
        task_names: dict[int, str] = {},
        seed: int | None = None,
    ):
        """
        Minimal bipedal walker wrapper for distillation training.

        The only purpose of this wrapper is to expose certain configurable
        environmental variables, such as hull starting positions and command
        velocities.

        Upon every step and reset call, the relevant command information will
        be returned in the info dict under ["cmd"], such as the current state's sampled command
        velocity. The observation returned will be the basic proprioceptive data
        collected by bipedalWalker (i.e., no lidar), which means you should not
        use this along with another ProprioObsWrapper!

        Since this environment is purely for distillation purposes, the reward
        function will be hardcoded to return 0 (not like you use it anyways).

        Args:
        env: The BipedalWalker environment to wrap.
        ep_time: Maximum episode duration in seconds. Default: 10.
        tasks: Specifies what tasks there are. Purely cosmetic in rendering. Default: []
        """
        super().__init__(env)

        self._rng = np.random.default_rng(seed)

        # specified here: https://github.com/openai/gym/blob/bc212954b6713d5db303b3ead124de6cba66063e/gym/envs/box2d/bipedal_walker.py#L30
        self._FPS = 50

        # for determining truncation
        self._max_steps: int = ep_time * self._FPS
        self._step_count: int = 0

        # hull reset sampling — configurable via config_hull_reset
        self._hull_x_range: tuple[float, float] = (0.0, 0.0)
        self._hull_y_range: tuple[float, float] = (0.0, 0.0)
        self._hull_rot_range: tuple[float, float] = (-0.2, 0.2)
        self._hull_vel_x_range: tuple[float, float] = (-0.2, 0.2)
        self._hull_vel_y_range: tuple[float, float] = (0.0, 0.0)

        # joint reset sampling — configurable via config_joint_reset
        self._hip_range: tuple[float, float] = (-0.3, 0.3)
        self._knee_range: tuple[float, float] = (-0.3, -0.1)
        self._joint_vel_range: tuple[float, float] = (-0.2, 0.2)

        # cmd vel — configurable via config_cmd_vel
        self._vel_switch_steps: int = self._max_steps
        self._vel_sample_range: tuple[float, float] = (0.0, 0.0)
        self._vel_sample_zero: float = 0.0
        self._cmd_vel_interp_step: float = 5.0 / self._FPS  # max per-step tracking delta

        # runtime cmd vel state
        self._cmd_vel: float = 0.0
        self._cmd_vel_target: float = 0.0

        # cmd tilt — configurable via config_cmd_tilt
        self._tilt_switch_steps: int = self._max_steps
        self._tilt_sample_range: tuple[float, float] = (0.0, 0.0)
        self._tilt_sample_zero: float = 0.0
        self._cmd_tilt_interp_step: float = 1.0 / self._FPS  # max per-step tracking delta

        # runtime cmd tilt state
        self._cmd_tilt: float = 0.0
        self._cmd_tilt_target: float = 0.0

        # tasks (purely cosmetic)
        self._task_names = task_names
        self._cur_task_id = 0

        # active task bits for rendering: [walk_active, hop_active, tilt_active]
        # None = no conditional arrow rendering (used during training)
        self._active_tasks: list[int] | None = None

    def set_task(self, id):
        self._cur_task_id = id

    def config_hull_reset(
        self,
        x_range: tuple[float, float] = (0.0, 40.0),
        y_range: tuple[float, float] = (0.2, 0.3),
        rot_range: tuple[float, float] = (-0.2, 0.2),
        vel_x_range: tuple[float, float] = (-0.2, 0.2),
        vel_y_range: tuple[float, float] = (0.0, 0.0),
    ):
        """Configure hull initial-condition sampling at each reset.

        Args:
            x_range: Range added to the hull's default spawn x. Spread this out to
                randomize where along the terrain the agent starts. Default: (0.0, 40.0).
            y_range: Range added to the hull's default spawn y. The hull is already
                snapped above the terrain, so small offsets give a slight drop-in.
                Default: (0.0, 0.0).
            rot_range: Range for the hull's initial tilt. Full limits are approx ±1.0
                Default: (-0.2, 0.2).
            vel_x_range: Range for the hull's initial horizontal velocity.
                Default: (-0.2, 0.2).
            vel_y_range: Range for the hull's initial vertical velocity.
                Default: (0.0, 0.0).
        """
        self._hull_x_range = x_range
        self._hull_y_range = y_range
        self._hull_rot_range = rot_range
        self._hull_vel_x_range = vel_x_range
        self._hull_vel_y_range = vel_y_range

    def config_joint_reset(
        self,
        hip_range: tuple[float, float] = (-0.3, 0.3),
        knee_range: tuple[float, float] = (-0.3, -0.1),
        joint_vel_range: tuple[float, float] = (-0.2, 0.2),
    ):
        """Configure leg joint initial-condition sampling at each reset.

        Args:
            hip_range: Range for each hip joint angle. Hard limits are (-0.8, 1.1).
                Default: (-0.3, 0.3).
            knee_range: Range for each knee joint angle. Hard limits are (-1.6, -0.1);
                must stay negative (bent). Default: (-0.3, -0.1).
            joint_vel_range: Range for all joint angular velocities.
                Default: (-0.2, 0.2).
        """
        self._hip_range = hip_range
        self._knee_range = knee_range
        self._joint_vel_range = joint_vel_range

    def config_cmd_vel(
        self,
        sample_range: tuple[float, float] = (-2.5, 2.5),
        switch_time: float = 5.0,
        interp_speed: float = 5.0,
        zero_prob: float = 0.15,
    ):
        """Configure command velocity sampling and switching behaviour.

        Args:
            sample_range: Range from which a new target velocity is drawn at each
                switch. Default: (-2.5, 2.5).
            switch_time: How often (seconds) a new target velocity is sampled.
                Set to ep_time or higher to keep velocity constant per episode.
                Default: 5.0.
            interp_speed: Max rate (units/sec) at which the live command tracks the
                target; clamped per step to interp_speed/FPS. Higher = snappier.
                Default: 5.0.
            zero_prob: Probability [0, 1] of sampling exactly 0 instead of drawing
                from sample_range. Default: 0.2.
        """
        self._vel_sample_range = sample_range
        self._vel_switch_steps = max(int(switch_time * self._FPS), 1)
        self._cmd_vel_interp_step = interp_speed / self._FPS
        self._vel_sample_zero = zero_prob

    def config_cmd_tilt(
        self,
        sample_range: tuple[float, float] = (-0.75, 0.75),
        switch_time: float = 10.0,
        interp_speed: float = 1.0,
        zero_prob: float = 0.15,
    ):
        """Configure command tilt sampling and switching behaviour.

        Args:
            sample_range: Range from which a new target tilt angle is drawn at each
                switch. Default: (-0.75, 0.75).
            switch_time: How often (seconds) a new target tilt is sampled.
                Set to ep_time or higher to keep tilt constant per episode.
                Default: 10.0.
            interp_speed: Max rate (units/sec) at which the live command tracks the
                target; clamped per step to interp_speed/FPS. Higher = snappier.
                Default: 1.0.
            zero_prob: Probability [0, 1] of sampling exactly 0 instead of drawing
                from sample_range. Default: 0.15.
        """
        self._tilt_sample_range = sample_range
        self._tilt_switch_steps = max(int(switch_time * self._FPS), 1)
        self._cmd_tilt_interp_step = interp_speed / self._FPS
        self._tilt_sample_zero = zero_prob

    def set_active_tasks(self, task_one_hot: list[int] | None):
        """Set which tasks are active for rendering purposes.

        Args:
            task_one_hot: [walk_active, hop_active, tilt_active] — determines which
                arrow overlays are drawn. None disables conditional rendering entirely.
                Called from play scripts; not needed during training.
        """
        self._active_tasks = task_one_hot

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # step environment
        obs, _, term, trunc, info = super().step(action)

        # detect truncation
        self._step_count += 1
        trunc = trunc or self._step_count >= self._max_steps

        # change command velocity if specified
        if (
            self._vel_switch_steps < self._max_steps
            and self._step_count % self._vel_switch_steps == 0
        ):
            if self._rng.random() > self._vel_sample_zero:
                self._cmd_vel_target = self._rng.uniform(*self._vel_sample_range)
            else:
                self._cmd_vel_target = 0.0

        # track the target with a per-step slew-rate clamp
        delta_vel = self._cmd_vel_target - self._cmd_vel
        self._cmd_vel += float(
            np.clip(delta_vel, -self._cmd_vel_interp_step, self._cmd_vel_interp_step)
        )

        # change command tilt if specified
        if (
            self._tilt_switch_steps < self._max_steps
            and self._step_count % self._tilt_switch_steps == 0
        ):
            if self._rng.random() > self._tilt_sample_zero:
                self._cmd_tilt_target = self._rng.uniform(*self._tilt_sample_range)
            else:
                self._cmd_tilt_target = 0.0

        # track the target with a per-step slew-rate clamp
        delta_tilt = self._cmd_tilt_target - self._cmd_tilt
        self._cmd_tilt += float(
            np.clip(delta_tilt, -self._cmd_tilt_interp_step, self._cmd_tilt_interp_step)
        )

        info["cmd"] = {
            "x_vel": self._cmd_vel,
            "tilt": self._cmd_tilt,
        }

        return obs, 0, term, trunc, info  # no reward information needed

    def _draw_velocity_arrows(self, env):
        unwrapped: Any = self.unwrapped
        real_vel_x: float = unwrapped.hull.linearVelocity.x

        def _draw_arrow(pygame, env, vel_x: float, color: tuple, y_offset: int = 0):
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

            sign = math.copysign(1.0, vel_x)  # direction
            mag = abs(vel_x) * ARROW_SCALE * 10  # pixel length of shaft

            ex = sx + sign * mag
            ey = sy

            # shorten shaft so it ends at base of head, not tip
            shaft_ex = ex - sign * HEAD_LEN
            pygame.draw.line(
                env.surf, color, (int(sx), int(sy)), (int(shaft_ex), int(ey)), 2
            )

            # constant-size arrowhead: tip at (ex, ey), base perpendicular in screen y
            p1 = (int(ex), int(ey))
            p2 = (int(ex - sign * HEAD_LEN), int(ey - HEAD_WIDTH))
            p3 = (int(ex - sign * HEAD_LEN), int(ey + HEAD_WIDTH))

            pygame.draw.polygon(env.surf, color, [p1, p2, p3])

        # green = command
        _draw_arrow(pygame, env, self._cmd_vel, color=(9, 176, 12), y_offset=-10)
        # blue = real
        _draw_arrow(pygame, env, real_vel_x, color=(71, 126, 255))

    def _draw_tilt_arrows(self, env):
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

        bx, by = screen_dir(hull_ang)
        anchor = (cx + HULL_FRONT_OFFSET * bx, cy + HULL_FRONT_OFFSET * by)
        blue_end = (anchor[0] + BLUE_ARROW_LEN * bx, anchor[1] + BLUE_ARROW_LEN * by)

        gx, gy = screen_dir(self._cmd_tilt)
        green_end = (anchor[0] + (BLUE_ARROW_LEN + GREEN_ARROW_LEN) * gx, anchor[1] + (BLUE_ARROW_LEN + GREEN_ARROW_LEN) * gy)

        draw_segment(anchor, blue_end, (71, 126, 255))   # blue = actual hull angle
        draw_segment(anchor, green_end, (9, 176, 12))    # green = command tilt

    def _draw_task_info(self, env):
        unwrapped: Any = self.unwrapped
        real_vel_x: float = unwrapped.hull.linearVelocity.x

        SCALE = 30.0
        MARGIN = 10.0

        pygame.font.init()
        font = pygame.font.SysFont("Courier New", 16, bold=True)

        task_name = self._task_names.get(self._cur_task_id, "NA")
        lines = [
            f"task:  {task_name}",
            f"cmd_x_vel:    {self._cmd_vel:+.2f}",
            f"cmd_tilt:     {self._cmd_tilt:+.2f}",
            f"active_tasks: {self._active_tasks}",
            f"hull_x_vel:   {real_vel_x:+.2f}",
        ]

        scroll_x = int(getattr(env, "scroll", 0) * SCALE) + MARGIN
        y = MARGIN
        for line in lines:
            surf = font.render(line, True, (255, 0, 0))
            env.surf.blit(surf, (scroll_x, y))
            y += surf.get_height() + 2

    def render(self):
        result = super().render()  # gets rgb_array frame with base rendering done

        env: Any = self.unwrapped
        if not hasattr(env, "surf") or env.surf is None:
            return result

        if self._active_tasks is not None:
            if self._active_tasks[0] or self._active_tasks[1]:  # walk or hop active
                self._draw_velocity_arrows(env)
            if self._active_tasks[2]:  # tilt active
                self._draw_tilt_arrows(env)

        self._draw_task_info(env)

        # re-grab the frame after drawing on surf
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(env.surf)), axes=(1, 0, 2)
        )[:, -600:]

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict[str, Any]]:
        self._step_count = 0
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = super().reset(seed=seed, options=options)

        # resample command velocity
        if self._rng.random() > self._vel_sample_zero:
            self._cmd_vel = self._rng.uniform(*self._vel_sample_range)
        else:
            self._cmd_vel = 0.0

        self._cmd_vel_target = self._cmd_vel  # live starts at target (no startup lag)

        # resample command tilt
        if self._rng.random() > self._tilt_sample_zero:
            self._cmd_tilt = self._rng.uniform(*self._tilt_sample_range)
        else:
            self._cmd_tilt = 0.0

        self._cmd_tilt_target = self._cmd_tilt  # live starts at target (no startup lag)

        env: Any = self.unwrapped
        hull = env.hull
        legs = env.legs

        # all of these constants are defined here:
        # https://github.com/openai/gym/blob/bc212954b6713d5db303b3ead124de6cba66063e/gym/envs/box2d/bipedal_walker.py#L31
        VIEWPORT_H = 400
        SCALE = 30.0
        LEG_DOWN = -8 / SCALE
        LEG_H = 34 / SCALE

        hull.position += b2Vec2(
            self._rng.uniform(*self._hull_x_range),
            self._rng.uniform(*self._hull_y_range),
        )

        hull_x = env.hull.position.x
        ground_y = float(np.interp(hull_x, env.terrain_x, env.terrain_y))
        ground_y_rel = ground_y - VIEWPORT_H / SCALE / 4
        # move hull to above ground
        hull.position.y += ground_y_rel

        hull.linearVelocity += b2Vec2(
            self._rng.uniform(*self._hull_vel_x_range),
            self._rng.uniform(*self._hull_vel_y_range),
        )
        hull.angle += self._rng.uniform(*self._hull_rot_range)

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

            hip_angle = self._rng.uniform(*self._hip_range)
            knee_angle = self._rng.uniform(*self._knee_range)

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
                body.angularVelocity = self._rng.uniform(*self._joint_vel_range)
                body.awake = True

        # apply the changes
        obs = self.step(np.array([0, 0, 0, 0]))[0]

        # add command info
        info["cmd"] = {
            "x_vel": self._cmd_vel,
            "tilt": self._cmd_tilt,
        }

        return obs, info
