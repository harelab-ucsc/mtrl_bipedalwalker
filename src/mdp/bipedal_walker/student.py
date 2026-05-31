import numpy as np
import torch
from torch import nn

BASE_OBS_SIZE = 14
OBS_SIZE_V2 = (
    BASE_OBS_SIZE + 2 + 3
)  # base obs + cmd_vel + cmd_tilt + 3 one-hot task bits (walk=100, flamingo=010, tilt=001)
ACT_SIZE = 4

# Student actor mirrors the ppo_bc actor trunk + head (see
# scripts/ppo_bc/train_config.py::TrainConfig.hidden_dims and
# src/ppo_bc_sb3/common/policies.py::PpoBcNetwork): Linear -> ELU per hidden dim,
# then a final Linear head to the action dim (no output activation).
HIDDEN_BC = (512, 256, 256, 128, 64)


class StudentModel(nn.Module):
    def __init__(
        self,
        obs_size: int = OBS_SIZE_V2,
        act_size: int = ACT_SIZE,
        hidden: tuple = HIDDEN_BC,
    ):
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
    def obs(
        base_obs: np.ndarray,
        task_id: int,
        cmd_vel: float,
        cmd_tilt: float,
        task_bit_override: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        task_bits = task_bit_override

        if task_bits is None:  # determine task bits implicitly
            if task_id == 0 or task_id == 1:
                task_bits = [1, 0, 0]  # walk
            elif task_id == 2:
                task_bits = [0, 1, 0]  # flamingo
            else:
                task_bits = [0, 0, 1]  # body_tilt

        return np.concatenate([base_obs, [cmd_vel, cmd_tilt], task_bits])
