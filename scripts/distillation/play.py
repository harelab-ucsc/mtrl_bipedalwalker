import os
import warnings
from gymnasium import make
import numpy as np
import pygame
from stable_baselines3 import PPO

from pynput import keyboard
from pynput.keyboard import Key, KeyCode

import torch
from utils.paths import MODELS_DIR
from wrappers.bipedal_walker.distill_env import DistillEnv
from mdp.bipedal_walker.student import (
    StudentModelMLV2,
    OBS_SIZE_V2,
    ACT_SIZE,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================================

EXPERIMENT_NAME = "distill/ml"
MODEL_CHECKPOINT = "best.pt"

# =========================================

_sim_paused = False
_sim_step = False
_sim_res = False
_sim_task_delta = 0
_starting_seed = 420


def main():
    global _sim_paused, _sim_step, _sim_res, _sim_task_delta, _starting_seed

    # start key listeners
    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # start to listen on a separate thread

    print(f'=== Starting experiment "{EXPERIMENT_NAME}" ===')

    # load env
    print("Loading environments...")
    env = make("BipedalWalker-v3", render_mode="rgb_array")

    env = DistillEnv(
        env,
        ep_time=15,
        task_names={
            0: "walk forward",
            1: "walk back",
            2: "hop forward",
            3: "hop back",
            4: "body tilt",
        },
    )

    env.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=10, interp_time=1, zero_prob=0.15)
    
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    clock = pygame.time.Clock()
    
    def configureEnv(e: DistillEnv, task_id: int):
        # config the env to a certain task
        e.set_task(task_id)
        if task_id == 0 or task_id == 2:  # walk / hop forward
            e.config_hull_reset(x_range=(0.0, 40.0), y_range=(0.2, 0.3))
            e.config_cmd_vel(sample_range=(0.0, 5.0), interp_time=1, switch_time=3)
            e.set_active_tasks([1, 0, 0] if task_id == 0 else [0, 1, 0])
        elif task_id == 1 or task_id == 3:  # walk / hop backward
            e.config_hull_reset(x_range=(40.0, 80.0), y_range=(0.2, 0.3))
            e.config_cmd_vel(sample_range=(-5.0, 0.0), interp_time=1, switch_time=3)
            e.set_active_tasks([1, 0, 0] if task_id == 1 else [0, 1, 0])
        elif task_id == 4:  # body tilt
            e.config_hull_reset(x_range=(20.0, 60.0))
            e.set_active_tasks([0, 0, 1])
    
    configureEnv(env, 0)
    obs, info = env.reset(seed=_starting_seed)
    cmd_vel = info["cmd"]["x_vel"]
    cmd_tilt = info["cmd"]["tilt"]
    task_id = 0

    # load model
    print(f'Loading model "{MODEL_CHECKPOINT}"...')
    model_path = MODELS_DIR / EXPERIMENT_NAME / MODEL_CHECKPOINT
    model = StudentModelMLV2()

    model.to("cpu")
    model.load_state_dict(torch.load(model_path, weights_only=False)["policy"])
    model.eval()

    # manually render
    def render():
        frame = env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))  # type: ignore
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        clock.tick(50)

    with torch.no_grad():
        while 1:
            pygame.event.pump()  # keep window alive on pause

            if _sim_task_delta != 0:
                task_id = (task_id + _sim_task_delta) % 5
                _sim_task_delta = 0
                _sim_res = True

            if _sim_res:
                configureEnv(env, task_id)
                obs, info = env.reset(seed=_starting_seed)
                if _starting_seed is not None:
                    _starting_seed += 1
                cmd_vel = info["cmd"]["x_vel"]
                cmd_tilt = info["cmd"]["tilt"]

                _sim_res = False
                render()

                continue

            if _sim_paused:
                if not _sim_step:
                    continue
                else:
                    _sim_step = False

            assert env.action_space.shape is not None

            obs_s = StudentModelMLV2.obs(obs, task_id, cmd_vel, cmd_tilt)
            action = model(torch.tensor(obs_s, dtype=torch.float32))
            obs, _, term, trunc, info = env.step(action)
            cmd_vel = info["cmd"]["x_vel"]
            cmd_tilt = info["cmd"]["tilt"]

            render()

            if term or trunc:
                _sim_res = True


def on_press(key: Key | KeyCode | None) -> None:
    global _sim_paused, _sim_step, _sim_res, _sim_task_delta

    if isinstance(key, KeyCode):
        k = key.char
    elif isinstance(key, Key):
        k = key.name
    else:
        return

    if k == "space":
        _sim_paused = not _sim_paused
        print("Paused" if _sim_paused else "Resumed")
    elif k == "s":
        _sim_step = True
    elif k == "r":
        _sim_res = True
    elif k == "w":
        _sim_task_delta = -1
    elif k == "e":
        _sim_task_delta = 1
    elif k == "q":
        print("Exiting...")
        os._exit(0)


if __name__ == "__main__":
    main()
