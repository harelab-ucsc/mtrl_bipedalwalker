import numpy as np
from stable_baselines3 import PPO
import torch
from utils.paths import MODELS_DIR

BASE_OBS_SIZE = 14
ACT_SIZE = 4

EXPERT_MODEL_PATHS = [
    "experts/walk_forward",
    "experts/walk_backward",
    "experts/hop_forward",
    "experts/hop_backward",
]

class HybridModel():
    def __init__(self):
        self.expert_models = [
            PPO.load(MODELS_DIR / i, env=None, device="cpu") for i in EXPERT_MODEL_PATHS
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cmd_vel = x[-3]
        task_id = 0
        if cmd_vel >= 0:  # forward task
            task_id = 0 if x[-2] == 1 else 2
        else:  # backward task
            task_id = 1 if x[-2] == 1 else 3
        
        action, _ = self.expert_models[task_id].predict(x[:-2].numpy())
        return torch.tensor(action)

    @staticmethod
    def obs(base_obs: np.ndarray, task_id: int, cmd_vel: float) -> np.ndarray:
        task_spec = [1, 0] if task_id < 2 else [0, 1]  # walk=10, hop=01
        return np.concatenate([base_obs, [cmd_vel], task_spec])
    