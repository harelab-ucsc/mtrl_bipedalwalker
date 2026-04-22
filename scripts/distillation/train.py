# https://arxiv.org/pdf/2505.11164

import os
import warnings

from gymnasium import Space, make
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm
from utils.paths import MODELS_DIR, LOGS_DIR, ROOT
from wrappers.bipedal_walker.distill_env import DistillEnv

EXPERT_MODEL_PATHS = [
    "experts/walk_forward",
    "experts/walk_backward",
    "experts/hop_forward",
    "experts/hop_backward",
]

_sim_paused = False
_sim_step = False
_sim_res = False


def main():
    global _sim_paused, _sim_step, _sim_res

    print("Loading environments...")

    env = make("BipedalWalker-v3", render_mode="rgb_array")
    env = DistillEnv(
        env,
        ep_time=10,
        tasks={1: "walk forward", 2: "walk backward", 3: "hop forward", 4: "hop backward"},
    )

    # load expert models
    print("Loading experts...")
    EXPERT_MODELS = [
        PPO.load(MODELS_DIR / i, env=None, device="cpu") for i in EXPERT_MODEL_PATHS
    ]

    BASE_OBS_SIZE = 14
    OBS_SIZE = (
        BASE_OBS_SIZE + 3
    )  # one for command velocity, 2 one hot encoded for task specification.
    ACT_SIZE = 4

    # create student model
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type  # type: ignore
    else:
        device = "cpu"

    class StudentModel(nn.Module):
        def __init__(self, obs_size: int, act_size: int):
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

    student = StudentModel(OBS_SIZE, ACT_SIZE).to(device)

    # helper for DAgger
    def configureEnv(
        env: DistillEnv, task_id: int
    ):
        reset_x_range = None
        cmd_vel_sample_range = None
        cmd_vel_interp_range = None

        if task_id == 0 or task_id == 2:  # walk / hop forward
            reset_x_range = (0.0, 40.0)
            cmd_vel_sample_range = (0.0, 5.0)
            cmd_vel_interp_range = 0.5
        elif task_id == 1 or task_id == 3:  # walk / hop backward
            reset_x_range = (40.0, 80.0)
            cmd_vel_sample_range = (-5.0, 0.0)
            cmd_vel_interp_range = 0.5

        # safety check
        assert reset_x_range is not None
        assert cmd_vel_sample_range is not None
        assert cmd_vel_interp_range is not None

        # set configuration
        env.config_hull_reset(
            x_range=reset_x_range
        )
        env.config_cmd_vel(
            sample_range=cmd_vel_sample_range,
            interp_time=cmd_vel_interp_range
        )
    
    def forwardExpert(obs: np.ndarray, task_id: int, cmd_vel: float):
        if 0 <= task_id < 4:  # locomotion task, append cmd velocity
            obs = np.append(obs, cmd_vel)
        
        action, _ = EXPERT_MODELS[task_id].predict(obs, deterministic=True)
        return action
    
    def forwardStudent(obs: np.ndarray, task_id: int, cmd_vel: float):
        obs = np.append(obs, cmd_vel)  # add cmd velocity
        
        # create one hot encoded task specification
        task_spec = [0, 0]
        if task_id == 0 or task_id == 1:
            task_spec[0] = 1
        elif task_id == 2 or task_id == 3:
            task_spec[1] = 1
        obs = np.append(obs, task_spec)

        input = torch.from_numpy(obs.astype("float32")).to(device)
        
        action = student.forward(input)
        return action.to("cpu")

    # DAgger
    T = 500  # T-step trajectory
    batch = 5  # total batch size = batch * parallel env ct
    epoch = 300  # test

    D = []  # aggregated dataset D

    bar = tqdm(total=epoch * batch * T, desc="Training", ascii=" ░▒█")
    for _ in range(epoch):
        # 1. get dataset D_i = {(s, π*(s))} of visited state by π_i and action by expert
        Di = []

        with torch.no_grad():
            # 1. collect trajectories
            for _ in range(batch):
                # init random task and setup env
                current_task = np.random.choice(list(range(4)))
                configureEnv(env, current_task)
                obs, info = env.reset()
                cmd_vel = info["cmd"]["x_vel"]

                for _ in range(T):
                    # forward prop the appropriate expert & student
                    act_expert = forwardExpert(obs, current_task, cmd_vel)
                    act_student = forwardStudent(obs, current_task, cmd_vel)
                    # step env
                    obs, _, trunc, term, info = env.step(act_student)
                    done = trunc or term
                    # collect data
                    Di.append((obs, act_expert))

                    # if done, pick a different task and reset
                    if done:
                        current_task = np.random.choice(list(range(4)))
                        configureEnv(env, current_task)
                        obs, info = env.reset()
                        cmd_vel = info["cmd"]["x_vel"]
                    
                    bar.update(1)

            # 2. aggregate dataset
            D += Di

            # 3. train student on D
            obs_list, act_list = zip(*D)
            obs_arr = np.array(obs_list)   # (N, BASE_OBS_SIZE)
            act_arr = np.array(act_list)   # (N, ACT_SIZE)

    bar.close()


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="__array_wrap__ must accept context and return_scalar arguments",
        category=DeprecationWarning,
    )
    main()
