import numpy as np
import torch
from torch import nn

BASE_OBS_SIZE = 14
OBS_SIZE_V2 = (
    BASE_OBS_SIZE + 2 + 3
)  # base obs + cmd_vel + cmd_tilt + 3 trailing task bits. The bits are scheme-
# dependent (see mdp.bipedal_walker.tasks): gait (two_leg, one_leg, 0) by default,
# or legacy one-hot (walk, flamingo, tilt). The dim is identical either way.
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
        cmd_vel: float,
        cmd_tilt: float,
        task_bits,
    ) -> np.ndarray:
        """Assemble the student obs: ``[base_obs, cmd_vel, cmd_tilt, *task_bits]``.

        ``task_bits`` is the 3-element trailing identifier — under the gait scheme
        the task's ``gait_bits`` ``(two_leg, one_leg, 0)``; under legacy onehot the
        one-hot ``(walk, flamingo, tilt)``. The caller owns the scheme, so this just
        concatenates whatever bits it's handed.
        """
        return np.concatenate([base_obs, [cmd_vel, cmd_tilt], task_bits])
