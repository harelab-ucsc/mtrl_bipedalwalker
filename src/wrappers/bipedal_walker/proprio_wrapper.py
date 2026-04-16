import numpy as np
from gymnasium import spaces, Wrapper

_LIDAR_START = 14
_LIDAR_END = 24  # exclusive — 10 lidar rays at indices [14, 24)


class ProprioObsWrapper(Wrapper):
    """
    Strips the 10 lidar rays (indices 14-23) from any BipedalWalker-based
    environment, leaving only proprioceptive observations.

    Works with any wrapper that produces a flat Box observation of at least 24
    elements where indices [14:24] are the lidar rays. Trailing elements after
    the lidar block (e.g. the cmd_vel appended by WalkReward / HopReward) are
    preserved.

    Observation layout before stripping (e.g. with cmd_vel):
        [0:14]   proprioceptive  (hull_ang … leg_2_contact)
        [14:24]  10 lidar rays   ← removed
        [24]     cmd_vel         ← shifted to index 14

    Observation layout after stripping:
        [0:14]   proprioceptive
        [14:]    any trailing elements (empty for envs without cmd_vel)
    """

    def __init__(self, env):
        super().__init__(env)
        base = self.observation_space
        assert isinstance(base, spaces.Box), (
            "ProprioObsWrapper requires a Box observation space"
        )
        assert base.shape[0] >= _LIDAR_END, (
            f"Expected obs dim ≥ {_LIDAR_END}, got {base.shape[0]}"
        )
        keep = np.r_[:_LIDAR_START, _LIDAR_END:base.shape[0]]
        self.observation_space = spaces.Box(
            low=base.low[keep],
            high=base.high[keep],
            dtype=base.dtype,  # type: ignore
        )

    def _strip_lidar(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs[:_LIDAR_START], obs[_LIDAR_END:]])

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        return self._strip_lidar(obs), rew, term, trunc, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return self._strip_lidar(obs), info