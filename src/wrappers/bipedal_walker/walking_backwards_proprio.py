import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from wrappers.bipedal_walker.walking_backwards_env import WalkBackReward


_PROPRIO_SLICE = slice(0, 14)   # hull + joint obs
_LIDAR_SLICE   = slice(14, 24)  # 10 lidar rays

class ProprioWalkBackReward(WalkBackReward):
    """
    Extends WalkReward by stripping the 10 lidar rays from the observation.
    Resulting obs layout (15 dims):
        [0:14]  proprioceptive  (hull_ang … leg_2_contact)
        [14]    cmd_vel
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

        # WalkReward already built a 25-dim Box. Rebuild it as 15-dim
        base = self.observation_space  # the 25-dim Box from WalkReward
        keep = np.r_[_PROPRIO_SLICE, -1]  # indices [0..13, 24]

        self.observation_space = spaces.Box(
            low=base.low[keep], # type: ignore
            high=base.high[keep], # type: ignore
            dtype=base.dtype, # type: ignore
        )

    @staticmethod
    def _strip_lidar(obs: np.ndarray) -> np.ndarray:
        # obs is 25-dim from WalkReward: proprio | lidar | cmd_vel
        return np.concatenate([obs[_PROPRIO_SLICE], obs[-1:]])

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        return self._strip_lidar(obs), rew, term, trunc, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return self._strip_lidar(obs), info
