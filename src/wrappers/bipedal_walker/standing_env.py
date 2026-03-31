from typing import Any, SupportsFloat

from gymnasium import Env, Wrapper
from gymnasium.core import ActType, ObsType

import numpy as np

from mdp.bipedal_walker.rewards import body_lin_vel_l2, leg_contact


class StandReward(Wrapper):
    """
    Wrapper that replaces BipedalWalker's default reward with a standing reward.
    """

    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

    def _compute_stand_rew(
        self,
        obs: np.ndarray,
        terminated: bool
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
            ("hull_ang_l2", hull_ang_l2, -0.4),  # penalize deviation from 0
            ("down_firing_lidar", down_firing_lidar, 0.3),
            ("termination", termination, -10.0),
        ]

        components = {name: float(r * w) for name, r, w in rewards_cfg}
        return sum(components.values()), components

    def step(
        self,
        action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs: np.ndarray
        term: bool
        obs, rew, term, trunc, info = super().step(action)

        rew, info["reward_terms"] = self._compute_stand_rew(obs, term)

        return (obs, rew, term, trunc, info)
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        
        for leg in self.unwrapped.legs:
            # TODO: add joint randomization here
            pass
        
        return obs, info
