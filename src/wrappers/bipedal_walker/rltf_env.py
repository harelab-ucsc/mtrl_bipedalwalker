from typing import Any, Literal, SupportsFloat, cast

from gymnasium import Env, Wrapper
from gymnasium.core import ActType, ObsType
from Box2D import b2Vec2

import numpy as np
import math

from gymnasium import spaces
import pygame

from gymnasium.envs.box2d.bipedal_walker import BipedalWalker
from wrappers.bipedal_walker.proprio_wrapper import ProprioObsWrapper

from mdp.bipedal_walker.rl_finetune_rewards import hop_rew, walk_rew

Landscapes = Literal["walk", "hop"]
LandscapeConfig = dict[Landscapes, tuple[float, float]]

class RlFTEnv(ProprioObsWrapper):
    def __init__(
        self,
        env: Env[ObsType, ActType],
        ep_time: int = 20,
        vel_switching_freq: float = 3,
        task_switching_freq: float = 6,
        vel_interp_speed: float = 0.5,
        vel_sample_range: tuple[float, float] = (-5.0, 5.0),
        vel_sample_zero: float = 0.2,
        hull_x_range: tuple[float, float] = (20.0, 60.0),
        landscape_correction: LandscapeConfig = {},  # set initial mean / variance corrections for each reward landscape
    ):
        """
        Wrapper for performing RL distillation, as described here:
        https://arxiv.org/pdf/2505.11164

        Unlike the distillation env, where the training script controls task sampling
        and task configuration, this env will control task sampling. The main reason
        is that this env needs to compute corresponding rewards for each tasks, and
        doing so in the training script with multiple environments using SubprocVecEnv
        can get really annoying.
        """
        super().__init__(env)

        # specified here: https://github.com/openai/gym/blob/bc212954b6713d5db303b3ead124de6cba66063e/gym/envs/box2d/bipedal_walker.py#L30
        FPS = 50

        # environmental and training setups
        self._max_steps: int = ep_time * FPS
        self._step_count: int = 0

        # velocity & task sampling
        self._task_switch_steps: int = np.floor(task_switching_freq * FPS)
        self._vel_switch_steps: int = np.floor(vel_switching_freq * FPS)
        self._vel_sample_range: tuple[float, float] = vel_sample_range
        self._vel_sample_zero: float = vel_sample_zero
        self._hull_x_range: tuple[float, float] = hull_x_range

        # velocity and task commands
        self._cmd_vel: float = 0
        self._cmd_vel_target: float = 0
        self._cmd_vel_buf: list[float] = [0]  # smooth out transitions
        self._max_vel_buf_size: int = np.floor(
            vel_interp_speed * FPS
        )  # smooth out transitions
        self._cmd_task_id: int = 0  # 0 = walk, 1 = hop
        
        # reward landscape correction terms
        self._landscape_correction = landscape_correction

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

        # configure observation space to fit the new cmds
        base = self.observation_space
        self.observation_space = spaces.Box(
            low=np.concatenate([base.low, [-np.inf, 0.0, 0.0]]),  # type: ignore
            high=np.concatenate([base.high, [np.inf, 1.0, 1.0]]),  # type: ignore
            dtype=np.float64,
        )
        
    def apply_landscape_correction(self, landscape: Landscapes, mean: float, var: float):
        self._landscape_correction[landscape] = (mean, var)

    def _compute_walk_reward(
        self, obs: np.ndarray, terminated: bool
    ) -> tuple[SupportsFloat, dict[str, float], dict[str, float], dict[str, float]]:
        return walk_rew(
            cast(BipedalWalker, self.unwrapped),
            obs,
            self._cmd_vel,
            terminated,
            prev_vel_x=self._prev_vel_x,
            prev_vel_y=self._prev_vel_y,
            prev_accel_x=self._prev_accel_x,
            prev_accel_y=self._prev_accel_y,
        )

    def _compute_hop_reward(
        self, obs: np.ndarray, terminated: bool
    ) -> tuple[SupportsFloat, dict[str, float], dict[str, float], dict[str, float]]:
        r, r_terms, r_raws, r_weights, state_update = hop_rew(
            cast(BipedalWalker, self.unwrapped),
            obs,
            self._cmd_vel,
            terminated,
            last_leg_contact=self._last_leg_contact,
            last_obs_8=self._last_obs_8,
            last_obs_13=self._last_obs_13,
            steps_since_hop=self._steps_since_hop,
        )
        # update hop state
        self._last_leg_contact = state_update["last_leg_contact"]
        self._last_obs_8 = state_update["last_obs_8"]
        self._last_obs_13 = state_update["last_obs_13"]
        self._steps_since_hop = state_update["steps_since_hop"]

        return r, r_terms, r_raws, r_weights

    def _derive_full_obs(
        self, base_obs: np.ndarray, cmd_vel: float, task_id: int
    ) -> np.ndarray:
        task_spec = np.eye(2)[task_id]  # walk=10, hop=01
        return np.concatenate([base_obs, [cmd_vel], task_spec])

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        env: BipedalWalker = cast(BipedalWalker, self.unwrapped)
        assert env.hull, "cannot find env.hull — environment may be broken!"

        pre_vel_x = env.hull.linearVelocity.x
        pre_vel_y = env.hull.linearVelocity.y

        # step environment
        base_obs, _, term, trunc, info = super().step(action)

        # detect truncation
        self._step_count += 1
        trunc = trunc or self._step_count >= self._max_steps

        # compute reward + term infos based on task_id
        if self._cmd_task_id == 0:  # compute walk reward
            rew, info["reward_terms"], info["reward_raw"], info["reward_weights"] = (
                self._compute_walk_reward(base_obs, term)
            )
        else:
            rew, info["reward_terms"], info["reward_raw"], info["reward_weights"] = (
                self._compute_hop_reward(base_obs, term)
            )
            
        # apply corrective terms when specified
        task_rew_raw = rew
        walk_cfg = self._landscape_correction.get("walk")
        hop_cfg = self._landscape_correction.get("hop")
        if walk_cfg is not None and self._cmd_task_id == 0:
            rew = (float(rew) - walk_cfg[0]) / (walk_cfg[1] ** 0.5 + 1e-8)
        elif hop_cfg is not None and self._cmd_task_id == 1:
            rew = (float(rew) - hop_cfg[0]) / (hop_cfg[1] ** 0.5 + 1e-8)

        # prefix all reward terms for logging
        task_suffix = "W" if self._cmd_task_id == 0 else "H"
        info["reward_terms"] = {
            f"{k}_{task_suffix}": v for k, v in info["reward_terms"].items()
        }
        info["reward_raw"] = {
            f"{k}_{task_suffix}": v for k, v in info["reward_raw"].items()
        }
        info["reward_weights"] = {
            f"{k}_{task_suffix}": v for k, v in info["reward_weights"].items()
        }
        info["task"] = self._cmd_task_id
        info["task_reward_raw"] = task_rew_raw

        # update jerk + accel tracking state
        post_vel_x = env.hull.linearVelocity.x
        post_vel_y = env.hull.linearVelocity.y
        self._prev_accel_x = post_vel_x - pre_vel_x
        self._prev_accel_y = post_vel_y - pre_vel_y
        self._prev_vel_x = post_vel_x
        self._prev_vel_y = post_vel_y

        # resample command velocity if specified
        if (
            self._vel_switch_steps < self._max_steps
            and self._step_count % self._vel_switch_steps == 0
        ):
            if np.random.random() > self._vel_sample_zero:
                self._cmd_vel_target = np.random.uniform(*self._vel_sample_range)
            else:
                self._cmd_vel_target = 0.0

        # push the target command to the buf
        self._cmd_vel_buf.append(self._cmd_vel_target)
        if len(self._cmd_vel_buf) > self._max_vel_buf_size:
            # trim front
            self._cmd_vel_buf.pop(0)
        # update cmd vel
        self._cmd_vel = float(np.mean(self._cmd_vel_buf))

        # resample task if specified
        if (
            self._task_switch_steps < self._max_steps
            and self._step_count % self._task_switch_steps == 0
        ):
            self._cmd_task_id = np.random.choice([0, 1])

        # augment base_obs to include task + cmd velocity
        obs = self._derive_full_obs(base_obs, self._cmd_vel, self._cmd_task_id)

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

        # green = command
        _draw_arrow(env, self._cmd_vel, color=(9, 176, 12), y_offset=-10)
        # blue = real
        _draw_arrow(env, real_vel_x, color=(71, 126, 255))

    def _draw_task_info(self, env):
        unwrapped: Any = self.unwrapped
        real_vel_x: float = unwrapped.hull.linearVelocity.x

        SCALE = 30.0
        MARGIN = 10.0

        pygame.font.init()
        font = pygame.font.SysFont("Courier New", 16, bold=True)

        task_name = "NA"
        if self._cmd_vel == 0:
            task_name = "walk @ 0" if self._cmd_task_id == 0 else "hop @ 0"
        elif self._cmd_vel > 0:
            task_name = "walk forward" if self._cmd_task_id == 0 else "hop forward"
        else:
            task_name = "walk backward" if self._cmd_task_id == 0 else "hop backward"

        lines = [
            f"task:  {task_name}",
            f"cmd_x_vel: {self._cmd_vel:+.2f}",
            f"hull_x_vel: {real_vel_x:+.2f}",
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

        self._draw_velocity_arrows(env)
        self._draw_task_info(env)

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

        self._last_leg_contact = -1
        self._last_obs_8 = 0.0
        self._last_obs_13 = 0.0
        self._steps_since_hop = 0

        obs, info = super().reset(seed=seed, options=options)

        # change command velocity
        if np.random.random() > self._vel_sample_zero:
            self._cmd_vel = np.random.uniform(*self._vel_sample_range)
        else:
            self._cmd_vel = 0.0
        # resample task
        self._cmd_task_id = np.random.choice([0, 1])
        info["task"] = self._cmd_task_id  # set info

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

        # append command velocity to observations
        obs = self._derive_full_obs(obs, self._cmd_vel, self._cmd_task_id)

        return obs, info
