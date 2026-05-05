import os
import warnings
from gymnasium import make
import numpy as np
import pygame
from pynput import keyboard
from pynput.keyboard import Key, KeyCode

import torch
from utils.paths import MODELS_DIR
from wrappers.bipedal_walker.distill_env import DistillEnv
from mdp.bipedal_walker.student import (
    StudentModelXS,
    StudentModelS,
    StudentModelM,
    StudentModelML,
    StudentModelL,
    StudentModelXL,
    StudentModelXLL,
    StudentModelXLLL,
    StudentModel,
    OBS_SIZE,
    ACT_SIZE,
)
from mdp.bipedal_walker.hybrid import HybridModel

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================================

EXPERIMENTS = [
    ("distill/ml", "ML (70,324)", StudentModelML),
    ("distill/xl", "XL (173,956)", StudentModelXL),
    ("distill/m", "M (46,020)", StudentModelM),
    ("hybrid", "Hybrid (MoE)", HybridModel),
]

MODEL_CHECKPOINT = "best.pt"

ENV_W, ENV_H = 600, 400
MAX_COLS = 2

# =========================================

_sim_paused = False
_sim_step = False
_sim_res = False
_sim_task_delta = 0


def main():
    global _sim_paused, _sim_step, _sim_res, _sim_task_delta

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print(f'=== Comparison: {[label for _, label, _ in EXPERIMENTS]} ===')

    print("Loading environments...")
    envs: list[DistillEnv] = []
    for _ in EXPERIMENTS:
        raw = make("BipedalWalker-v3", render_mode="rgb_array")
        envs.append(DistillEnv(
            raw,
            ep_time=15,
            task_names={
                0: "walk forward",
                1: "walk back",
                2: "hop forward",
                3: "hop back",
            },
            seed=420,
        ))

    print("Loading models...")
    models: list[StudentModel] = []
    for exp_name, label, model_class in EXPERIMENTS:
        m = model_class()
        if exp_name != "hybrid":
            # hybrid model is not a pytorch nn, and the experts are loaded in the class
            m.to("cpu")
            m.load_state_dict(torch.load(MODELS_DIR / exp_name / MODEL_CHECKPOINT, weights_only=False)["policy"])
            m.eval()
        models.append(m)
        print(f'  Loaded "{label}"')

    pygame.init()
    pygame.font.init()
    n_cols = min(len(envs), MAX_COLS)
    n_rows = -(-len(envs) // MAX_COLS)  # ceil division
    screen = pygame.display.set_mode((ENV_W * n_cols, ENV_H * n_rows))
    clock = pygame.time.Clock()
    label_font = pygame.font.SysFont("Courier New", 32, bold=True)

    def configureEnv(e: DistillEnv, task_id: int):
        if task_id == 0 or task_id == 2:  # walk / hop forward
            x_range = (0.0, 40.0)
            vel_range = (0.0, 5.0)
        else:  # walk / hop backward
            x_range = (40.0, 80.0)
            vel_range = (-5.0, 0.0)
        e.set_task(task_id)
        e.config_hull_reset(x_range=x_range, y_range=(0.2, 0.3))
        e.config_cmd_vel(sample_range=vel_range, interp_time=1, switch_time=3)

    task_id = 0
    obs_list: list[np.ndarray] = []
    cmd_vels: list[float] = []
    done_list: list[bool] = []
    reset_seed = 420

    def reset_all(tid: int):
        nonlocal reset_seed
        obs_list.clear()
        cmd_vels.clear()
        done_list.clear()
        for env in envs:
            configureEnv(env, tid)
            obs, info = env.reset(seed=reset_seed)
            obs_list.append(obs)
            cmd_vels.append(info["cmd"]["x_vel"])
            done_list.append(False)
        reset_seed += 1

    def render():
        for i, (env, (_, label, _)) in enumerate(zip(envs, EXPERIMENTS)):
            col, row = i % n_cols, i // n_cols
            x, y = col * ENV_W, row * ENV_H
            if done_list[i]:
                surf = pygame.Surface((ENV_W, ENV_H))
                surf.fill((0, 0, 0))
            else:
                frame = env.render()
                if frame is not None:
                    surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))  # type: ignore
                else:
                    surf = pygame.Surface((ENV_W, ENV_H))
                    surf.fill((0, 0, 0))
            screen.blit(surf, (x, y))
            lbl = label_font.render(label, True, (0, 0, 255))
            screen.blit(lbl, (x + ENV_W - lbl.get_width() - 15, y + 10))
        pygame.display.flip()
        clock.tick(50)

    reset_all(task_id)

    with torch.no_grad():
        while True:
            pygame.event.pump()  # keep window alive on pause

            if _sim_task_delta != 0:
                task_id = (task_id + _sim_task_delta) % 4
                _sim_task_delta = 0
                _sim_res = True

            if _sim_res:
                reset_all(task_id)
                _sim_res = False
                render()
                continue

            if _sim_paused:
                if not _sim_step:
                    continue
                else:
                    _sim_step = False

            for i, (env, model) in enumerate(zip(envs, models)):
                if done_list[i]:
                    continue
                obs_s = model.obs(obs_list[i], task_id, cmd_vels[i])
                action = model.forward(torch.tensor(obs_s, dtype=torch.float32))
                obs, _, term, trunc, info = env.step(action)
                obs_list[i] = obs
                cmd_vels[i] = info["cmd"]["x_vel"]
                if term or trunc:
                    done_list[i] = True

            render()

            if all(done_list):
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
