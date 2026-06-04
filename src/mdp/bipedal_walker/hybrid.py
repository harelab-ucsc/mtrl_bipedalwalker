import numpy as np
from stable_baselines3 import PPO
import torch
from utils.paths import MODELS_DIR
from mdp.bipedal_walker.tasks import GAIT

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
    """Legacy onehot oracle (V1 2-bit + V2 3-bit one-hot obs). Kept for old
    checkpoints; new gait runs use HybridModelV2(scheme="gait")."""

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
    """Oracle baseline for the V2 setup (obs: [base(14), cmd_vel, cmd_tilt, b0, b1, b2]).

    Each active component routes to its single-task expert; combined tasks (no
    combination expert) average the active experts' actions (a single-task obs has
    one active expert, so the average reduces to it). Scheme-aware:

    "gait" (default) — bits = (two_leg, one_leg, unused):
      one_leg → hop_forward (cmd_vel >= 0) / hop_backward (cmd_vel < 0)
      two_leg → body_tilt (cmd_tilt != 0) and/or walk_forward/backward (by cmd_vel
                sign); a pure stand (both commands 0) defaults to walk @ 0.
    "onehot" (legacy) — bits = (walk, flamingo, tilt):
      walk → walk_forward/backward; flamingo → hop_backward @ 0; tilt → body_tilt.
    """

    def __init__(self, scheme: str = GAIT):
        self.scheme = scheme
        self.expert_models = [
            PPO.load(MODELS_DIR / i, env=None, device="cpu") for i in EXPERT_MODEL_PATHS
        ]

    def _predict(self, idx: int, base_obs: np.ndarray, cmd: float) -> np.ndarray:
        return self.expert_models[idx].predict(
            np.append(base_obs, cmd), deterministic=True
        )[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_obs = x[:BASE_OBS_SIZE].numpy()
        cmd_vel  = x[-5].item()
        cmd_tilt = x[-4].item()
        b0, b1, b2 = x[-3].item(), x[-2].item(), x[-1].item()

        actions = []
        if self.scheme == GAIT:
            two_leg, one_leg = b0, b1
            if one_leg == 1:  # directional hop (flamingo = hop_forward @ 0)
                idx = 2 if cmd_vel >= 0 else 3
                actions.append(self._predict(idx, base_obs, cmd_vel))
            if two_leg == 1:
                if cmd_tilt != 0:  # tilt component
                    actions.append(self._predict(4, base_obs, cmd_tilt))
                if cmd_vel != 0 or cmd_tilt == 0:  # walk component (pure-stand → walk @ 0)
                    idx = 0 if cmd_vel >= 0 else 1
                    actions.append(self._predict(idx, base_obs, cmd_vel))
        else:  # onehot
            walk_bit, hop_bit, tilt_bit = b0, b1, b2
            if walk_bit == 1:
                actions.append(self._predict(0 if cmd_vel >= 0 else 1, base_obs, cmd_vel))
            if hop_bit == 1:
                actions.append(self._predict(3, base_obs, 0.0))
            if tilt_bit == 1:
                actions.append(self._predict(4, base_obs, cmd_tilt))

        return torch.tensor(np.mean(actions, axis=0))
