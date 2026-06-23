from typing import Any, Callable, Literal, Sequence, SupportsFloat, cast

from gymnasium import Env, Wrapper
from gymnasium.core import ActType, ObsType
from Box2D import b2Vec2

from mdp.bipedal_walker.rl_finetune_rewards import RewardState, compositional_rew
from mdp.bipedal_walker.tasks import (
    BY_NAME,
    BY_NAME_GAIT,
    GAIT,
    GaitTask,
    SINGLE_TASKS_GAIT,
    TaskSpec,
    coerce_task,
    constrain_vel_range,
    effective_cmd,
    resolve_single_task,
    reward_mode,
    _name_from_bits,
)
import numpy as np
import math

from gymnasium import spaces
import pygame

from gymnasium.envs.box2d.bipedal_walker import (
    BipedalWalker,
    TERRAIN_HEIGHT,
    TERRAIN_LENGTH,
    TERRAIN_STEP,
)
from wrappers.bipedal_walker.proprio_wrapper import ProprioObsWrapper


def _flat_generate_terrain(self, hardcore):
    """Drop-in for BipedalWalker._generate_terrain that builds a perfectly flat
    strip at TERRAIN_HEIGHT (no GRASS bump noise, no hardcore features). Bound onto
    the unwrapped env when RlFTEnv is constructed with flat_terrain=True, so every
    reset regenerates flat ground. Mirrors gymnasium's edge-building loop."""
    self.terrain, self.terrain_x, self.terrain_y = [], [], []
    for i in range(TERRAIN_LENGTH):
        self.terrain_x.append(i * TERRAIN_STEP)
        self.terrain_y.append(TERRAIN_HEIGHT)
    self.terrain_poly = []
    for i in range(TERRAIN_LENGTH - 1):
        poly = [
            (self.terrain_x[i], self.terrain_y[i]),
            (self.terrain_x[i + 1], self.terrain_y[i + 1]),
        ]
        self.fd_edge.shape.vertices = poly
        t = self.world.CreateStaticBody(fixtures=self.fd_edge)
        color = (76, 255 if i % 2 == 0 else 204, 76)
        t.color1 = t.color2 = color
        poly += [(poly[1][0], 0), (poly[0][0], 0)]
        self.terrain_poly.append((poly, (102, 153, 76)))
    self.terrain.reverse()


class RlFTEnv(ProprioObsWrapper):
    def __init__(
        self,
        env: Env[ObsType, ActType],
        ep_time: int = 15,
        cmd_switching_time: tuple[float, float] = (3.0, 4.0),
        task_switching_time: float = 6,
        task_switch_replacement: bool = False,
        cmd_interp_speed: tuple[float, float] = (5.0, 1.0),
        cmd_sample_range: tuple[tuple[float, float], tuple[float, float]] = (
            (-5.0, 5.0),
            (-0.75, 0.75),
        ),
        cmd_sample_zero: tuple[float, float] = (0.2, 0.15),
        allowed_task_mixing: Sequence[tuple[int, int, int] | TaskSpec | GaitTask] = (
            SINGLE_TASKS_GAIT
        ),
        use_rew_for_individual_tasks: bool = False,
        hull_x_range: tuple[float, float] = (20.0, 60.0),
        manual_ctrl_mode: bool = False,
        task_scheme: str = GAIT,
        flat_terrain: bool = False,
    ):
        """
        RL fine-tuning env that owns its own task and command sampling, unlike the
        distillation env where the training script drives that externally.

        Args:
            ep_time: Episode length in seconds.
            cmd_switching_time: How often (seconds) each command component is resampled.
                Format is (vel, tilt). Interpolation smooths the transition to the new target.
            task_switching_time: How often (seconds) the task (walk / flamingo / tilt) is resampled.
            task_switch_replacement: When False (default), task draws within an episode are without
                replacement (no back-to-back repeats; the adversarial PMF is renormalized over the
                remaining tasks). Raises at init if an episode would need more draws than there are
                allowed tasks. When True, every draw is independent (with replacement).
            cmd_interp_speed: Limits how fast the live command tracks the target to avoid
                jarring transitions. Format is (vel, tilt).
            cmd_sample_range: Uniform sample bounds for each command component. Format is
                ((vel_min, vel_max), (tilt_min, tilt_max)).
            cmd_sample_zero: Probability of sampling exactly zero for each component —
                (vel_p_zero, tilt_p_zero). Ensures the agent sees the stationary case.
            allowed_task_mixing: List of allowed task-flag combinations (walk, flamingo, tilt).
                Each entry is a 3-tuple of 0/1 flags. Task sampling picks uniformly from this
                list. Use single-flag rows for pure tasks (e.g. (1,0,0) = walk only) and
                multi-flag rows for combinations (e.g. (1,0,1) = walk while tilting). The
                same flags gate the command mask in `_effective_cmd_vec`, so downstream
                reward terms can be selected by indexing into this same flag schema.
            hull_x_range: Range (m) along the terrain from which the hull x-position is
                sampled at reset, so the agent trains across varied terrain patches.
            manual_ctrl_mode: When True, disables all command and task resampling so an
                external caller can drive commands directly (e.g. for play/eval scripts).
            task_scheme: "gait" (default) or "onehot" (legacy). Selects how the 3 obs
                bits are interpreted and how commands are sampled/masked. Under "gait"
                each row of allowed_task_mixing should be a mdp.bipedal_walker.tasks
                GaitTask (gait + command ranges); under "onehot" a TaskSpec or raw
                3-tuple. See mdp.bipedal_walker.tasks for the scheme semantics.
            flat_terrain: When True, force a perfectly flat terrain by disabling the
                GRASS height randomization (binds a flat _generate_terrain onto the
                underlying BipedalWalker). Default False keeps the bumpy terrain.
        """
        super().__init__(env)

        # Force perfectly flat ground by replacing the terrain generator on the
        # unwrapped env. The Box2D collision edges are built from terrain_y during
        # generation, so this must override _generate_terrain (called every reset)
        # rather than post-process terrain_y after the fact.
        if flat_terrain:
            import types

            self.unwrapped._generate_terrain = types.MethodType(
                _flat_generate_terrain, self.unwrapped
            )

        # which obs-bit scheme this env runs under (drives sampling/masking/routing)
        self._task_scheme = task_scheme

        # specified here: https://github.com/openai/gym/blob/bc212954b6713d5db303b3ead124de6cba66063e/gym/envs/box2d/bipedal_walker.py#L30
        FPS = 50

        # environmental and training setups
        self._max_steps: int = ep_time * FPS
        self._step_count: int = 0

        # velocity & task sampling — indexed [vel, tilt]
        self._task_switch_steps: int = np.floor(task_switching_time * FPS)
        self._cmd_switch_steps: tuple[int, int] = (
            np.floor(cmd_switching_time[0] * FPS),
            np.floor(cmd_switching_time[1] * FPS),
        )
        self._cmd_sample_range = cmd_sample_range
        self._cmd_sample_zero = cmd_sample_zero
        # Coerce each row to a TaskSpec. Raw 3-tuples become velocity-unconstrained
        # specs (pre-refactor behavior); pass mdp.bipedal_walker.tasks specs (e.g.
        # SINGLE_TASKS) to treat walk_forward / walk_backward as distinct tasks.
        self._allowed_tasks: list[TaskSpec | GaitTask] = [
            coerce_task(row) for row in allowed_task_mixing
        ]
        assert len(self._allowed_tasks) > 0, "allowed_task_mixing must be non-empty"

        # Without replacement we draw distinct tasks per episode, so an episode must
        # not request more draws than there are tasks. Draws/episode = 1 (reset pick)
        # + one per in-episode switch boundary = 1 + (max_steps // task_switch_steps)
        # when switching is enabled (task_switch_steps < max_steps).
        self._task_switch_replacement = task_switch_replacement
        if not task_switch_replacement and self._task_switch_steps < self._max_steps:
            draws_per_ep = 1 + int(self._max_steps // self._task_switch_steps)
            if draws_per_ep > len(self._allowed_tasks):
                raise ValueError(
                    f"task_switch_replacement=False needs >= {draws_per_ep} distinct "
                    f"tasks per episode (ep_time={ep_time}s, "
                    f"task_switching_time={task_switching_time}s => {draws_per_ep} draws) "
                    f"but only {len(self._allowed_tasks)} allowed tasks. Lengthen "
                    f"task_switching_time, shorten ep_time, add tasks, or set "
                    f"task_switch_replacement=True."
                )

        self._use_rew_for_individual_tasks = use_rew_for_individual_tasks
        self._hull_x_range: tuple[float, float] = hull_x_range

        # velocity and tilt commands
        self._cmd_vec: tuple[float, float] = (0, 0)  # x_vel, tilt
        self._cmd_vec_target: tuple[float, float] = (0, 0)
        self._cmd_interp_step: tuple[float, float] = (
            cmd_interp_speed[0] / FPS,
            cmd_interp_speed[1] / FPS,
        )

        # task specification. _task_id_vec is the 3 obs bits (onehot one-hot or
        # gait bits); _active_task carries the richer identity (vel_sign for onehot,
        # command ranges for gait) and is what command sampling consults. They stay
        # in sync via _activate_task.
        self._task_id_vec: tuple[int, int, int] = (0, 0, 0)
        self._active_task: TaskSpec | GaitTask = (
            GaitTask("idle", (0, 0), (0.0, 0.0), (0.0, 0.0))
            if task_scheme == GAIT
            else coerce_task((0, 0, 0))
        )
        self._task_sample_probs: tuple[float, ...] | None = None  # None = uniform sampling
        # task indices already drawn this episode (for without-replacement sampling);
        # cleared at the top of reset().
        self._used_task_idxs: set[int] = set()

        # previous hull velocities and accelerations for jerk calculation
        self._prev_vel_x: float = 0.0
        self._prev_vel_y: float = 0.0
        self._prev_accel_x: float = 0.0
        self._prev_accel_y: float = 0.0

        # leg contacts for hop reward calculation
        self._last_leg_contact = -1  # 0 -> left; 1 -> right; -1 -> unset
        self._last_obs_8 = 0.0
        self._last_obs_13 = 0.0
        self._steps_since_hop = 0
        self._gait_trace = 0.0  # decaying cadence-quality trace (see reward)

        # configure observation space to fit the new cmds
        base = self.observation_space
        self.observation_space = spaces.Box(
            low=np.concatenate([base.low, [-np.inf, -np.inf, 0.0, 0.0, 0.0]]),  # type: ignore
            high=np.concatenate([base.high, [np.inf, np.inf, 1.0, 1.0, 1.0]]),  # type: ignore
            dtype=np.float64,
        )

        self.manual_ctrl_mode = manual_ctrl_mode

        # When set, task sampling in step()/reset() is bypassed and `_task_id_vec`
        # is pinned to this value. Used by the algorithm's adversarial eval loop
        # to force each isolated task while still letting commands resample.
        self._forced_task: TaskSpec | GaitTask | None = None

    def _coerce_task(self, task) -> TaskSpec | GaitTask:
        """Accept a TaskSpec/GaitTask, a task name, or a raw 3-tuple → task spec.

        Names resolve against the active scheme's registry (gait → BY_NAME_GAIT,
        onehot → BY_NAME)."""
        if isinstance(task, str):
            return BY_NAME_GAIT[task] if self._task_scheme == GAIT else BY_NAME[task]
        return coerce_task(task)

    def set_forced_task(self, task) -> None:
        """Pin the task or clear with None. ``task`` may be a TaskSpec, a task
        name (e.g. "walk_backward"), or a raw 3-tuple of 0/1 flags (treated as
        velocity-unconstrained). Command resampling is otherwise unaffected —
        use ``manual_ctrl_mode`` to freeze commands too."""
        self._forced_task = None if task is None else self._coerce_task(task)

    def set_task_sample_probs(self, probs) -> None:
        """Setter so VecEnv.env_method can push a new sampling PMF into the
        inner RlFTEnv across SubprocVecEnv workers. VecEnv.set_attr would only
        touch the outer Monitor wrapper, leaving this attribute stale. The PMF
        must be ordered to match ``allowed_task_mixing``."""
        self._task_sample_probs = tuple(probs) if probs is not None else None

    def _sample_task(self) -> TaskSpec | GaitTask:
        """Pick the next task. Forced > sampling PMF > uniform fallback.

        The PMF (set via set_task_sample_probs) is used when its length matches
        the number of allowed tasks — e.g. the adversarial loop pushes a PMF
        over the directional single tasks. Otherwise tasks are drawn uniformly.

        When task_switch_replacement is False, draws are without replacement within
        an episode: indices already drawn (tracked in _used_task_idxs, cleared at
        reset) are excluded and the PMF is renormalized over the remaining tasks.
        The init preflight guarantees enough distinct tasks for one episode's draws,
        so the candidate pool is never empty here.
        """
        if self._forced_task is not None:
            return self._forced_task

        n = len(self._allowed_tasks)
        if self._task_switch_replacement:
            candidates = list(range(n))
        else:
            candidates = [i for i in range(n) if i not in self._used_task_idxs]
            if not candidates:  # safety net; preflight should prevent this
                self._used_task_idxs.clear()
                candidates = list(range(n))

        if (
            self._task_sample_probs is not None
            and len(self._task_sample_probs) == n
        ):
            probs = np.array(
                [self._task_sample_probs[i] for i in candidates], dtype=float
            )
            total = probs.sum()
            idx = int(
                np.random.choice(candidates, p=probs / total)
                if total > 0
                else np.random.choice(candidates)
            )
        else:
            idx = int(np.random.choice(candidates))

        if not self._task_switch_replacement:
            self._used_task_idxs.add(idx)
        return self._allowed_tasks[idx]

    def _activate_task(self, spec: TaskSpec | GaitTask) -> None:
        """Make ``spec`` the current task: sync the 3-bit obs vector from it.

        Reset the gait phase when the task *bits* change so the contact state
        machine doesn't carry a stale cadence across the switch (walk_forward ↔
        walk_backward share bits and keep their alternating cadence)."""
        if spec.task_id_vec != self._task_id_vec:
            self._reset_gait_phase()
        self._active_task = spec
        self._task_id_vec = spec.task_id_vec

    def _reset_gait_phase(self) -> None:
        """Clear the leg-contact state machine and cadence trace."""
        self._last_leg_contact = -1
        self._last_obs_8 = 0.0
        self._last_obs_13 = 0.0
        self._steps_since_hop = 0
        self._gait_trace = 0.0

    def _effective_vel_range(self) -> tuple[float, float]:
        """Velocity sampling range for the active task. Gait: the task's own
        cmd_vel_range. Onehot: the configured global range with the task's sign
        constraint folded in."""
        if self._task_scheme == GAIT:
            return cast(GaitTask, self._active_task).cmd_vel_range
        return constrain_vel_range(
            self._cmd_sample_range[0], cast(TaskSpec, self._active_task).vel_sign
        )

    def _effective_tilt_range(self) -> tuple[float, float]:
        """Tilt sampling range for the active task. Gait: the task's own
        cmd_tilt_range (so walk/hop tasks pin tilt to 0). Onehot: the configured
        global tilt range (masking zeroes it for non-tilt tasks downstream)."""
        if self._task_scheme == GAIT:
            return cast(GaitTask, self._active_task).cmd_tilt_range
        return self._cmd_sample_range[1]

    def _sample_vel_target(self) -> float:
        """Sample a velocity command target from the active task's range,
        honoring the zero-sampling probability."""
        if np.random.random() > self._cmd_sample_zero[0]:
            return float(np.random.uniform(*self._effective_vel_range()))
        return 0.0

    def _sample_tilt_target(self) -> float:
        """Sample a tilt command target from the active task's range, honoring
        the zero-sampling probability."""
        if np.random.random() > self._cmd_sample_zero[1]:
            return float(np.random.uniform(*self._effective_tilt_range()))
        return 0.0

    def _compute_task_mixing_reward(
        self, base_obs: np.ndarray, terminated: bool
    ) -> tuple[SupportsFloat, dict[str, float], dict[str, float], dict[str, float]]:
        cmd_walk, cmd_tilt = self._effective_cmd_vec()

        # Combined tasks have no single expert to drive BC, so they always need the
        # task-specific RL signal; resolve_task returns None for them (and for
        # legacy onehot combos like (1,0,1)). Individual tasks only get the task
        # reward when explicitly enabled — otherwise expert BC drives learning and
        # the RL term is the (always-on) regularization/stability layer.
        is_combined = (
            resolve_single_task(self._task_id_vec, cmd_walk, cmd_tilt, self._task_scheme) is None
        )
        enable_task_reward = is_combined or self._use_rew_for_individual_tasks

        # gait mode the reward conditions on (walk / hop / quiet), scheme-aware.
        mode = reward_mode(self._task_id_vec, cmd_walk, self._task_scheme)

        r, r_terms, r_raws, r_weights, state_update = compositional_rew(
            env=cast(BipedalWalker, self.unwrapped),
            base_obs=base_obs,
            terminated=terminated,
            state=RewardState(
                prev_vel_x=self._prev_vel_x,
                prev_vel_y=self._prev_vel_y,
                prev_accel_x=self._prev_accel_x,
                prev_accel_y=self._prev_accel_y,
                last_leg_contact=self._last_leg_contact,
                last_obs_8=self._last_obs_8,
                last_obs_13=self._last_obs_13,
                steps_since_hop=self._steps_since_hop,
                gait_trace=self._gait_trace,
            ),
            task_bits=self._task_id_vec,
            # cmd_walk / cmd_tilt are the conditioned tracking targets. Onehot masks
            # them by task bit; gait leaves them raw (each task's command ranges
            # already zero the irrelevant one), so either way they double as the
            # "hold 0" anchors (stationary → hold still, non-tilt → hold upright).
            cmd_vel=cmd_walk,
            cmd_tilt=cmd_tilt,
            mode=mode,
            enable_task_reward=enable_task_reward,
            weight_overrides=None,
        )
        # update gait / contact bookkeeping
        self._last_leg_contact = state_update.last_leg_contact
        self._last_obs_8 = state_update.last_obs_8
        self._last_obs_13 = state_update.last_obs_13
        self._steps_since_hop = state_update.steps_since_hop
        self._gait_trace = state_update.gait_trace

        return r, r_terms, r_raws, r_weights

    def _effective_cmd_vec(self) -> tuple[float, float]:
        """The command vector the policy/reward sees. Scheme-aware (see
        tasks.effective_cmd): onehot masks vel by the walk bit and tilt by the
        tilt bit; gait passes the raw commands through (each task's ranges already
        zero the irrelevant one)."""
        return effective_cmd(self._cmd_vec, self._task_id_vec, self._task_scheme)

    def _derive_full_obs(
        self,
        base_obs: np.ndarray,
        cmd: tuple[float, float],
        task_id_vec: tuple[int, int, int],
    ) -> np.ndarray:
        return np.concatenate([base_obs, cmd, task_id_vec])

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        env: BipedalWalker = cast(BipedalWalker, self.unwrapped)
        assert env.hull, "cannot find env.hull — environment may be broken!"

        # book keep previous velocities for acc and jerk calculations
        pre_vel_x = env.hull.linearVelocity.x
        pre_vel_y = env.hull.linearVelocity.y

        # step sim
        base_obs, _, term, trunc, info = super().step(action)

        # detect truncation
        self._step_count += 1
        trunc = trunc or self._step_count >= self._max_steps

        # compute just stability reward for now
        # adapt to dynamic reward switching in the future
        rew, info["reward_terms"], info["reward_raw"], info["reward_weights"] = (
            self._compute_task_mixing_reward(base_obs, term)
        )
        info["task"] = self._task_id_vec
        # directional task label, derived from the obs bits + effective commands so
        # it's correct even when _task_id_vec is driven externally (manual_ctrl_mode).
        # Combined tasks fall back to a composed name (e.g. "walk+tilt").
        _eff = self._effective_cmd_vec()
        _spec = resolve_single_task(self._task_id_vec, _eff[0], _eff[1], self._task_scheme)
        info["task_name"] = (
            _spec.name if _spec is not None else _name_from_bits(self._task_id_vec)
        )

        # update jerk + accel book keeping
        post_vel_x = env.hull.linearVelocity.x
        post_vel_y = env.hull.linearVelocity.y
        self._prev_accel_x = post_vel_x - pre_vel_x
        self._prev_accel_y = post_vel_y - pre_vel_y
        self._prev_vel_x = post_vel_x
        self._prev_vel_y = post_vel_y

        # resample velocity command if specified
        if (
            not self.manual_ctrl_mode
            and self._cmd_switch_steps[0] < self._max_steps
            and self._step_count % self._cmd_switch_steps[0] == 0
        ):
            self._cmd_vec_target = (self._sample_vel_target(), self._cmd_vec_target[1])

        # resample tilt command if specified
        if (
            not self.manual_ctrl_mode
            and self._cmd_switch_steps[1] < self._max_steps
            and self._step_count % self._cmd_switch_steps[1] == 0
        ):
            self._cmd_vec_target = (self._cmd_vec_target[0], self._sample_tilt_target())

        delta_vel = self._cmd_vec_target[0] - self._cmd_vec[0]
        delta_tilt = self._cmd_vec_target[1] - self._cmd_vec[1]
        self._cmd_vec = (
            self._cmd_vec[0]
            + float(
                np.clip(delta_vel, -self._cmd_interp_step[0], self._cmd_interp_step[0])
            ),
            self._cmd_vec[1]
            + float(
                np.clip(delta_tilt, -self._cmd_interp_step[1], self._cmd_interp_step[1])
            ),
        )

        # resample task on schedule unless manual control is on. _sample_task
        # honors _forced_task → _task_sample_probs (single-task mode only) →
        # uniform fallback. Mixed-task mode never uses the adversarial PMF;
        # add critic-based adversarial search there in the future.
        if (
            not self.manual_ctrl_mode
            and self._task_switch_steps < self._max_steps
            and self._step_count % self._task_switch_steps == 0
        ):
            self._activate_task(self._sample_task())
            # resample the velocity target under the new task's sign constraint
            # so e.g. a switch to walk_backward immediately heads backward.
            self._cmd_vec_target = (self._sample_vel_target(), self._cmd_vec_target[1])

        # augment base_obs with task-masked cmd vec + task id
        obs = self._derive_full_obs(
            base_obs, self._effective_cmd_vec(), self._task_id_vec
        )

        return obs, rew, term, trunc, info

    def _draw_velocity_arrows(self, env):
        unwrapped: Any = self.unwrapped
        real_vel_x: float = unwrapped.hull.linearVelocity.x

        def _draw_arrow(env, vel_x: float, color: tuple, y_offset: int = 0):
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

        # green = command velocity (post-mask, what the policy actually sees)
        _draw_arrow(env, self._effective_cmd_vec()[0], color=(9, 176, 12), y_offset=-10)
        # blue = real velocity
        _draw_arrow(env, real_vel_x, color=(71, 126, 255))

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
                env.surf,
                color,
                (int(start[0]), int(start[1])),
                (int(shaft_end[0]), int(shaft_end[1])),
                2,
            )
            px, py = -uy, ux
            tip = (int(end[0]), int(end[1]))
            p2 = (
                int(end[0] - ux * HEAD_LEN + px * HEAD_WIDTH),
                int(end[1] - uy * HEAD_LEN + py * HEAD_WIDTH),
            )
            p3 = (
                int(end[0] - ux * HEAD_LEN - px * HEAD_WIDTH),
                int(end[1] - uy * HEAD_LEN - py * HEAD_WIDTH),
            )
            pygame.draw.polygon(env.surf, color, [tip, p2, p3])

        bx, by = screen_dir(hull_ang)
        anchor = (cx + HULL_FRONT_OFFSET * bx, cy + HULL_FRONT_OFFSET * by)
        blue_end = (anchor[0] + BLUE_ARROW_LEN * bx, anchor[1] + BLUE_ARROW_LEN * by)

        gx, gy = screen_dir(self._effective_cmd_vec()[1])
        green_end = (
            anchor[0] + (BLUE_ARROW_LEN + GREEN_ARROW_LEN) * gx,
            anchor[1] + (BLUE_ARROW_LEN + GREEN_ARROW_LEN) * gy,
        )

        draw_segment(anchor, blue_end, (71, 126, 255))  # blue = actual hull angle
        draw_segment(anchor, green_end, (9, 176, 12))  # green = command tilt

    def _draw_task_info(self, env):
        unwrapped: Any = self.unwrapped
        real_vel_x: float = unwrapped.hull.linearVelocity.x
        real_tilt: float = unwrapped.hull.angle

        SCALE = 30.0
        MARGIN = 10.0

        pygame.font.init()
        font = pygame.font.SysFont("Courier New", 16, bold=True)

        # show the effective cmd — i.e. what the policy actually sees
        cmd_vel, cmd_tilt = self._effective_cmd_vec()

        # compose task name from the active gait/flags so combinations read naturally.
        parts: list[str] = []
        if self._task_scheme == GAIT:
            two_leg, one_leg = self._task_id_vec[0], self._task_id_vec[1]
            move = "forward" if cmd_vel > 0 else ("backward" if cmd_vel < 0 else None)
            if one_leg:
                parts.append(f"hop {move}" if move else "hop in place")
            elif two_leg:
                if move:
                    parts.append(f"walk {move}")
                if cmd_tilt != 0:
                    parts.append("tilt")
                if not parts:
                    parts.append("stand")
            # gait masks nothing → both commands are always live
            cmd_vel_str = f"{cmd_vel:+.2f}"
            cmd_tilt_str = f"{cmd_tilt:+.2f}"
        else:  # onehot: direction qualifier attaches to the walk component only
            task_vel, task_tilt = (self._task_id_vec[0], self._task_id_vec[2])
            if self._task_id_vec[0]:
                if cmd_vel == 0:
                    parts.append("walk @ 0")
                elif cmd_vel > 0:
                    parts.append("walk forward")
                else:
                    parts.append("walk backward")
            if self._task_id_vec[1]:
                parts.append("flamingo")
            if self._task_id_vec[2]:
                parts.append("tilt")
            cmd_vel_str = f"{cmd_vel:+.2f}" if task_vel else "DISABLED"
            cmd_tilt_str = f"{cmd_tilt:+.2f}" if task_tilt else "DISABLED"
        task_name = " + ".join(parts) if parts else "idle"

        lines = [
            f"task:  {task_name}",
            f"(real / cmd) x_vel: {real_vel_x:+.2f} / {cmd_vel_str}",
            f"(real / cmd) tilt:  {real_tilt:+.2f} / {cmd_tilt_str}",
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

        self._draw_task_info(env)

        # gait masks nothing, so both commands are always live → always draw both
        # arrows. Onehot only draws the arrow for an active task bit.
        if self._task_scheme == GAIT:
            self._draw_velocity_arrows(env)
            self._draw_tilt_arrows(env)
        else:
            if self._task_id_vec[0]:
                self._draw_velocity_arrows(env)
            if self._task_id_vec[2]:
                self._draw_tilt_arrows(env)

        # re-grab the frame after drawing on surf
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(env.surf)), axes=(1, 0, 2)
        )[:, -600:]

    def reset(self, *, seed=None, options=None) -> tuple[Any, dict[str, Any]]:
        self._step_count = 0
        self._prev_vel_x = 0.0
        self._prev_vel_y = 0.0
        self._prev_accel_x = 0.0
        self._prev_accel_y = 0.0

        self._reset_gait_phase()
        # start a fresh without-replacement pool for this episode
        self._used_task_idxs.clear()

        obs, info = super().reset(seed=seed, options=options)

        # sample task first (forced → PMF → uniform) so the velocity command can
        # respect the task's sign constraint. Skipped in manual control, where an
        # external caller drives _task_id_vec / _cmd_vec directly.
        if not self.manual_ctrl_mode:
            self._activate_task(self._sample_task())

        # sample initial command vec from the active task's ranges
        cmd_vel = self._sample_vel_target() if not self.manual_ctrl_mode else 0.0
        cmd_tilt = self._sample_tilt_target() if not self.manual_ctrl_mode else 0.0
        self._cmd_vec = (cmd_vel, cmd_tilt)
        self._cmd_vec_target = self._cmd_vec

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
        HULL_ROT_SAMPLE_LIM = (-0.2, 0.2)
        HULL_VEL_X_SAMPLE_LIM = (-0.2, 0.2)
        HULL_VEL_Y_SAMPLE_LIM = (0, 0)

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

        # apply the changes — call on unwrapped env, then strip lidar manually
        # since ProprioObsWrapper._strip_lidar is not applied when bypassing the wrapper chain
        obs = self._strip_lidar(env.step(np.array([0, 0, 0, 0]))[0])

        # append task-masked cmd vec + task id to observations
        obs = self._derive_full_obs(obs, self._effective_cmd_vec(), self._task_id_vec)

        return obs, info
