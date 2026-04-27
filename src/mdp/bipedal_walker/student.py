import numpy as np
import torch
from torch import nn


BASE_OBS_SIZE = 14
OBS_SIZE = BASE_OBS_SIZE + 3  # base obs + cmd_vel + 2 one-hot task bits (walk=10, hop=01)
ACT_SIZE = 4


class StudentModel(nn.Module):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, act_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(x)

    @staticmethod
    def obs(base_obs: np.ndarray, task_id: int, cmd_vel: float) -> np.ndarray:
        task_spec = [1, 0] if task_id < 2 else [0, 1]  # walk=10, hop=01
        return np.concatenate([base_obs, [cmd_vel], task_spec])
