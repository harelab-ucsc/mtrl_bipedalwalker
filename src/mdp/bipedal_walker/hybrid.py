import numpy as np
from stable_baselines3 import PPO
import torch
from utils.paths import MODELS_DIR

BASE_OBS_SIZE = 14
ACT_SIZE = 4

EXPERT_MODEL_PATHS = [
    "experts/walk_forward",   # 0
    "experts/walk_backward",  # 1
    "experts/hop_forward",    # 2
    "experts/hop_backward",   # 3
    "experts/body_tilt",      # 4
]


class HybridModel():
    def __init__(self):
        self.expert_models = [
            PPO.load(MODELS_DIR / i, env=None, device="cpu") for i in EXPERT_MODEL_PATHS
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_obs = x[:BASE_OBS_SIZE].numpy()

        if len(x) == 19:
            # V2 obs layout: [base(14), cmd_vel, cmd_tilt, walk, hop, tilt]
            cmd_vel  = x[-5].item()
            cmd_tilt = x[-4].item()
            walk_bit = x[-3].item()
            tilt_bit = x[-1].item()

            if tilt_bit == 1:
                action, _ = self.expert_models[4].predict(
                    np.append(base_obs, cmd_tilt), deterministic=True
                )
            elif cmd_vel >= 0:
                task_id = 0 if walk_bit == 1 else 2
                action, _ = self.expert_models[task_id].predict(
                    np.append(base_obs, cmd_vel), deterministic=True
                )
            else:
                task_id = 1 if walk_bit == 1 else 3
                action, _ = self.expert_models[task_id].predict(
                    np.append(base_obs, cmd_vel), deterministic=True
                )
        else:
            # V1 obs layout: [base(14), cmd_vel, walk, hop]
            cmd_vel  = x[-3].item()
            walk_bit = x[-2].item()

            if cmd_vel >= 0:
                task_id = 0 if walk_bit == 1 else 2
            else:
                task_id = 1 if walk_bit == 1 else 3

            action, _ = self.expert_models[task_id].predict(
                np.append(base_obs, cmd_vel), deterministic=True
            )

        return torch.tensor(action)

    @staticmethod
    def obs(base_obs: np.ndarray, task_id: int, cmd_vel: float) -> np.ndarray:
        task_spec = [1, 0] if task_id < 2 else [0, 1]  # walk=10, hop=01
        return np.concatenate([base_obs, [cmd_vel], task_spec])

    @staticmethod
    def obs_v2(base_obs: np.ndarray, task_id: int, cmd_vel: float, cmd_tilt: float) -> np.ndarray:
        if task_id < 2:    task_bits = [1, 0, 0]
        elif task_id < 4:  task_bits = [0, 1, 0]
        else:              task_bits = [0, 0, 1]
        return np.concatenate([base_obs, [cmd_vel, cmd_tilt], task_bits])


class HybridModelV2():
    """Oracle baseline for the V2 4-task setup.

    Routing per task (obs layout: [base(14), cmd_vel, cmd_tilt, walk, hop, tilt]):
      walk  [1,0,0] — walk_forward  expert (cmd_vel >= 0) or walk_backward (cmd_vel < 0)
      hop   [0,1,0] — hop_backward  expert @ 0  (flamingo is always hop_backward @ 0)
      tilt  [0,0,1] — body_tilt     expert with cmd_tilt
    """

    def __init__(self):
        self.expert_models = [
            PPO.load(MODELS_DIR / i, env=None, device="cpu") for i in EXPERT_MODEL_PATHS
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_obs = x[:BASE_OBS_SIZE].numpy()
        cmd_vel  = x[-5].item()
        cmd_tilt = x[-4].item()
        walk_bit = x[-3].item()
        hop_bit  = x[-2].item()
        tilt_bit = x[-1].item()

        if tilt_bit == 1:
            action, _ = self.expert_models[4].predict(
                np.append(base_obs, cmd_tilt), deterministic=True
            )
        elif hop_bit == 1:
            # flamingo: hop_backward expert at cmd_vel=0
            action, _ = self.expert_models[3].predict(
                np.append(base_obs, 0.0), deterministic=True
            )
        elif walk_bit == 1 and cmd_vel >= 0:
            action, _ = self.expert_models[0].predict(
                np.append(base_obs, cmd_vel), deterministic=True
            )
        elif walk_bit == 1 and cmd_vel < 0:
            action, _ = self.expert_models[1].predict(
                np.append(base_obs, cmd_vel), deterministic=True
            )

        return torch.tensor(action)
