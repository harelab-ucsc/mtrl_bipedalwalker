from typing import Any, SupportsFloat

from gymnasium import Env, Wrapper
from gymnasium.core import ActType, ObsType
from Box2D import b2Vec2

import numpy as np
import math

from mdp.bipedal_walker.rewards import body_lin_vel_l2, leg_contact


class StandReward(Wrapper):
    """
    Wrapper that replaces BipedalWalker's default reward with a standing reward.
    """

    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

    def _compute_stand_rew(
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

        Compute standing reward following these reward terms:
            - Minimize horizontal velocity
            - Minimize rotational velocity
            - Feet must be contacting ground
            - Must look straight forward
            - Maximize Lidar dist pointing straight down (first one)
            - Penalize termination
        """

        x_vel_l2 = body_lin_vel_l2([obs[2], obs[3]], y=False)
        hull_ang_vel = abs(obs[1]) ** 2
        leg_contacts = leg_contact([obs[8], obs[13]])
        hull_ang_l2 = obs[0] ** 2
        down_firing_lidar = obs[14]
        termination = 1 if terminated else 0

        # normalize some rewards
        hull_ang_l2 /= np.pi / 2  # from ±90deg -> ±1
        hull_ang_vel /= 1e-2  # emperically measured

        rewards_cfg: list[tuple[str, float, float]] = [
            ("x_vel_l2", x_vel_l2, -0.2),
            ("hull_ang_vel", hull_ang_vel, -0.2),  # penalize deviation from 0
            ("leg_contacts", leg_contacts, 0.1),
            ("hull_ang_l2", hull_ang_l2, -0.5),  # penalize deviation from 0
            ("down_firing_lidar", down_firing_lidar, 0.5),
            ("termination", termination, -20.0),
        ]

        components = {name: float(r * w) for name, r, w in rewards_cfg}
        return sum(components.values()), components

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs: np.ndarray
        term: bool
        obs, rew, term, trunc, info = super().step(action)

        rew, info["reward_terms"] = self._compute_stand_rew(obs, term)

        return (obs, rew, term, trunc, info)

    def reset(
        self, *, seed=None, options=None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        env = self.unwrapped
        hull = env.hull
        legs = env.legs
        
        # move hull up slightly to prevent legs from clipping into the ground
        hull.position += b2Vec2(0, 0.2)

        # all of these constants are defined here:
        # https://github.com/openai/gym/blob/bc212954b6713d5db303b3ead124de6cba66063e/gym/envs/box2d/bipedal_walker.py#L31
        SCALE = 30.0
        LEG_DOWN = -8 / SCALE
        LEG_H = 34 / SCALE

        # hip lim: (-0.8, 1.1)
        HIP_SAMPLE_LIM = (-0.5, 0.5)
        # knee lim: (-1.6, -0.1)
        KNEE_SAMPLE_LIM = (-0.5, -0.1)
        VEL_SAMPLE_LIM = (-1.0, 1.0)

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
                body.angularVelocity = np.random.uniform(*VEL_SAMPLE_LIM)
                body.awake = True

        # apply the changes
        obs = env.step(np.array([0, 0, 0, 0]))[0]
        return obs, info
