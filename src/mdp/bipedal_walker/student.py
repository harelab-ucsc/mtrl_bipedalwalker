import numpy as np
import torch
from torch import nn

BASE_OBS_SIZE = 14
OBS_SIZE = (
    BASE_OBS_SIZE + 3
)  # base obs + cmd_vel + 2 one-hot task bits (walk=10, hop=01)
OBS_SIZE_V2 = (
    BASE_OBS_SIZE + 2 + 3
)  # base obs + cmd_vel + cmd_tilt + 3 one-hot task bits (walk=100, hop=010, tilt=001)
ACT_SIZE = 4

HIDDEN_XS = (128, 64, 32)
HIDDEN_S = (192, 96, 48)
HIDDEN_M = (256, 128, 64)
HIDDEN_ML = (320, 160, 80)
HIDDEN_L = (384, 192, 96)
HIDDEN_XL = (512, 256, 128)
HIDDEN_XLL = (768, 384, 192)
HIDDEN_XLLL = (1024, 512, 256)


class StudentModel(nn.Module):
    def __init__(
        self,
        obs_size: int = OBS_SIZE,
        act_size: int = ACT_SIZE,
        hidden: tuple = HIDDEN_M,
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
    def obs(base_obs: np.ndarray, task_id: int, cmd_vel: float) -> np.ndarray:
        task_spec = [1, 0] if task_id < 2 else [0, 1]  # walk=10, hop=01
        return np.concatenate([base_obs, [cmd_vel], task_spec])


class StudentModelXS(StudentModel):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=HIDDEN_XS)


class StudentModelS(StudentModel):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=HIDDEN_S)


class StudentModelM(StudentModel):
    pass  # default hidden=HIDDEN_M


class StudentModelML(StudentModel):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=HIDDEN_ML)


class StudentModelL(StudentModel):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=HIDDEN_L)


class StudentModelXL(StudentModel):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=HIDDEN_XL)


class StudentModelXLL(StudentModel):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=HIDDEN_XLL)


class StudentModelXLLL(StudentModel):
    def __init__(self, obs_size: int = OBS_SIZE, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=HIDDEN_XLLL)


class StudentModelV2(nn.Module):
    def __init__(
        self,
        obs_size: int = OBS_SIZE_V2,
        act_size: int = ACT_SIZE,
        hidden: tuple = HIDDEN_M,
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


class StudentModelXSV2(StudentModelV2):
    def __init__(self, obs_size: int = OBS_SIZE_V2, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=HIDDEN_XS)


class StudentModelSV2(StudentModelV2):
    def __init__(self, obs_size: int = OBS_SIZE_V2, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=HIDDEN_S)


class StudentModelMV2(StudentModelV2):
    pass  # default hidden=HIDDEN_M


class StudentModelMLV2(StudentModelV2):
    def __init__(self, obs_size: int = OBS_SIZE_V2, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=HIDDEN_ML)


class StudentModelLV2(StudentModelV2):
    def __init__(self, obs_size: int = OBS_SIZE_V2, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=HIDDEN_L)


class StudentModelXLV2(StudentModelV2):
    def __init__(self, obs_size: int = OBS_SIZE_V2, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=HIDDEN_XL)


class StudentModelXLLV2(StudentModelV2):
    def __init__(self, obs_size: int = OBS_SIZE_V2, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=HIDDEN_XLL)


class StudentModelXLLLV2(StudentModelV2):
    def __init__(self, obs_size: int = OBS_SIZE_V2, act_size: int = ACT_SIZE):
        super().__init__(obs_size, act_size, hidden=HIDDEN_XLLL)
