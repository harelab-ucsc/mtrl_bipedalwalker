import os
from gymnasium import make
import numpy as np
import pygame
from stable_baselines3 import PPO

from pynput import keyboard
from pynput.keyboard import Key, KeyCode

import torch
from utils.paths import MODELS_DIR
from wrappers.bipedal_walker.distill_env import DistillEnv
from mdp.bipedal_walker.student import StudentModel

# =========================================

# EXPERIMENT_NAME = "distill/2-16_12_47-2026_04_27"
# EXPERIMENT_NAME = "distill/2_1-16_27_47-2026_04_27"
# EXPERIMENT_NAME = "distill/3-16_37_45-2026_04_27"
# EXPERIMENT_NAME = "distill/4-21_51_40-2026_04_27"
# EXPERIMENT_NAME = "distill/4_1-22_22_28-2026_04_27"
# EXPERIMENT_NAME = "distill/4_2-22_34_06-2026_04_27"
# EXPERIMENT_NAME = "distill/4_3-22_57_05-2026_04_27"
# EXPERIMENT_NAME = "distill/5-14_28_18-2026_04_28"
EXPERIMENT_NAME = "distill/5_1-14_41_02-2026_04_28"
# EXPERIMENT_NAME = "distill/5_2-14_41_15-2026_04_28"
# EXPERIMENT_NAME = "distill/5_3-15_03_06-2026_04_28"
# EXPERIMENT_NAME = "distill/5_4-15_11_49-2026_04_28"
# EXPERIMENT_NAME = "distill/5_5-15_12_03-2026_04_28"
# EXPERIMENT_NAME = "distill/5_6-15_12_22-2026_04_28"

MODEL_CHECKPOINT = "best.pt"

# =========================================

_sim_paused = False
_sim_step = False
_sim_res = False
_sim_task_delta = 0


def main():
    global _sim_paused, _sim_step, _sim_res, _sim_task_delta

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
        tasks={
            0: "walk forward",
            1: "walk back",
            2: "hop forward",
            3: "hop back",
        }
    )
    
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    clock = pygame.time.Clock()
    
    def configureEnv(e: DistillEnv, task_id: int):
        # config the env to a certain task
        if task_id == 0 or task_id == 2:  # walk / hop forward
            x_range = (0.0, 40.0)
            vel_range = (0.0, 5.0)
        else:  # walk / hop backward
            x_range = (40.0, 80.0)
            vel_range = (-5.0, 0.0)
        e.set_task(task_id)
        e.config_hull_reset(x_range=x_range, y_range=(0.2, 0.3))
        e.config_cmd_vel(sample_range=vel_range, interp_time=0.5, switch_time=3)
    
    configureEnv(env, 0)
    obs, info = env.reset()
    cmd_x_vel = info["cmd"]["x_vel"]
    task_id = 0

    # load model
    print(f'Loading model "{MODEL_CHECKPOINT}"...')
    model_path = MODELS_DIR / EXPERIMENT_NAME / MODEL_CHECKPOINT
    model = StudentModel()
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
                task_id = (task_id + _sim_task_delta) % 4
                _sim_task_delta = 0
                _sim_res = True

            if _sim_res:
                configureEnv(env, task_id)
                obs, info = env.reset()
                cmd_x_vel = info["cmd"]["x_vel"]

                _sim_res = False
                render()

                continue

            if _sim_paused:
                if not _sim_step:
                    continue
                else:
                    _sim_step = False

            assert env.action_space.shape is not None

            # append command to model input
            # obs = np.append(obs, cmd_x_vel)
            
            obs_s = model.obs(obs, task_id, cmd_x_vel)
            action = model(torch.tensor(obs_s, dtype=torch.float32))
            obs, _, term, trunc, info = env.step(action)
            cmd_x_vel = info["cmd"]["x_vel"]  # update command

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
