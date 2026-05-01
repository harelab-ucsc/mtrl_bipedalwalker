import numpy as np
import torch
from torch import nn


BASE_OBS_SIZE = 14
OBS_SIZE = BASE_OBS_SIZE + 3  # base obs + cmd_vel + 2 one-hot task bits (walk=10, hop=01)
ACT_SIZE = 4


class StudentModel(nn.Module):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE, hidden: tuple = (256, 128, 64)):
        super().__init__()
        layers = []
        in_dim = obs_size
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_size))
        self.policy = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(x)

    @staticmethod
    def obs(base_obs: np.ndarray, task_id: int, cmd_vel: float) -> np.ndarray:
        task_spec = [1, 0] if task_id < 2 else [0, 1]  # walk=10, hop=01
        return np.concatenate([base_obs, [cmd_vel], task_spec])


class StudentModelXS(StudentModel):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=(128, 64, 32))


class StudentModelS(StudentModel):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=(192, 96, 48))


class StudentModelM(StudentModel):
    pass  # default hidden=(256, 128, 64)


class StudentModelML(StudentModel):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=(320, 160, 80))


class StudentModelL(StudentModel):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=(384, 192, 96))


class StudentModelXL(StudentModel):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=(512, 256, 128))


class StudentModelXLL(StudentModel):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=(768, 384, 192))


class StudentModelXLLL(StudentModel):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=(1024, 512, 256))