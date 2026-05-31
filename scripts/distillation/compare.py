import os
import warnings
from gymnasium import make
import numpy as np
import pygame
from pynput import keyboard
from pynput.keyboard import Key, KeyCode

import torch
from utils.paths import rudin_distill_ckpt
from wrappers.bipedal_walker.distill_env import DistillEnv
from mdp.bipedal_walker.student import StudentModel

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================================
# models to compare side-by-side — each is a rudin[_adv]/distill/{version} run.
# spec: (label, adversarial, version, mix, checkpoint)

MODELS = [
    ("adv 1.0.0 nomix", True,  "1.0.0", False, "best.pt"),
    ("adv 1.1.0 nomix", True,  "1.1.0", False, "best.pt"),
    ("unif 1.0.0",      False, "1.0.0", True,  "best.pt"),
]

TASK_NAMES = {
    0: "walk forward",
    1: "walk backward",
    2: "flamingo",
    3: "body tilt",
    4: "walk forward + flamingo",
    5: "walk backward + flamingo",
}
N_TASKS = 6

ENV_W, ENV_H = 600, 400
MAX_COLS = 2
START_SEED = 420

# =========================================

_sim_paused = False
_sim_step = False
_sim_res = False
_sim_task_delta = 0


def configure_env(e: DistillEnv, task_id: int, mix: bool):
    """Config one env to a task; returns the 3-bit active-task vector for rendering."""
    e.set_task(task_id)

    if task_id == 0:  # walk forward
        e.config_hull_reset(x_range=(0.0, 40.0))
        e.config_cmd_vel(sample_range=(0.0, 5.0), interp_time=0.5, switch_time=3, zero_prob=0.2)
        e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15) if mix else e.config_cmd_tilt(zero_prob=1)
        active = (1, 0, 0)
    elif task_id == 1:  # walk backward
        e.config_hull_reset(x_range=(40.0, 80.0))
        e.config_cmd_vel(sample_range=(-5.0, 0.0), interp_time=0.5, switch_time=3, zero_prob=0.2)
        e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15) if mix else e.config_cmd_tilt(zero_prob=1)
        active = (1, 0, 0)
    elif task_id == 2:  # flamingo
        e.config_hull_reset(x_range=(20.0, 60.0))
        if mix:
            e.config_cmd_vel(sample_range=(-5.0, 5.0), switch_time=3, interp_time=0.5, zero_prob=0.2)
            e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
        else:
            e.config_cmd_vel(zero_prob=1)
            e.config_cmd_tilt(zero_prob=1)
        active = (0, 1, 0)
    elif task_id == 3:  # tilt
        e.config_hull_reset(x_range=(20.0, 60.0))
        e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
        e.config_cmd_vel(sample_range=(-5.0, 5.0), switch_time=3, interp_time=0.5, zero_prob=0.2) if mix else e.config_cmd_vel(zero_prob=1)
        active = (0, 0, 1)
    elif task_id == 4:  # walk forward + flamingo
        e.config_hull_reset(x_range=(0.0, 40.0))
        e.config_cmd_vel(sample_range=(0.0, 5.0), interp_time=0.5, switch_time=3, zero_prob=0.2)
        e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15) if mix else e.config_cmd_tilt(zero_prob=1)
        active = (1, 1, 0)
    else:  # task_id == 5: walk backward + flamingo
        e.config_hull_reset(x_range=(40.0, 80.0))
        e.config_cmd_vel(sample_range=(-5.0, 0.0), interp_time=0.5, switch_time=3, zero_prob=0.2)
        e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15) if mix else e.config_cmd_tilt(zero_prob=1)
        active = (1, 1, 0)

    e.set_active_tasks(list(active))
    return active


def main():
    global _sim_paused, _sim_step, _sim_res, _sim_task_delta

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print(f"=== Comparison: {[label for label, *_ in MODELS]} ===")

    print("Loading environments...")
    envs: list[DistillEnv] = []
    for _ in MODELS:
        raw = make("BipedalWalker-v3", render_mode="rgb_array")
        envs.append(DistillEnv(raw, ep_time=15, task_names=TASK_NAMES, seed=START_SEED))

    print("Loading models...")
    models: list[StudentModel] = []
    for label, adversarial, version, _mix, ckpt in MODELS:
        path = rudin_distill_ckpt(adversarial, version, ckpt)
        m = StudentModel()
        m.to("cpu")
        m.load_state_dict(torch.load(path, weights_only=False)["policy"])
        m.eval()
        models.append(m)
        print(f'  Loaded "{label}"  <-  {path}')

    pygame.init()
    pygame.font.init()
    n_cols = min(len(envs), MAX_COLS)
    n_rows = -(-len(envs) // MAX_COLS)  # ceil division
    screen = pygame.display.set_mode((ENV_W * n_cols, ENV_H * n_rows))
    clock = pygame.time.Clock()
    label_font = pygame.font.SysFont("Courier New", 32, bold=True)

    task_id = 0
    obs_list: list[np.ndarray] = []
    cmd_vels: list[float] = []
    cmd_tilts: list[float] = []
    active_bits: list[tuple] = []
    done_list: list[bool] = []
    reset_seed = START_SEED

    def reset_all(tid: int):
        nonlocal reset_seed
        obs_list.clear(); cmd_vels.clear(); cmd_tilts.clear()
        active_bits.clear(); done_list.clear()
        for env, (_, _, _, mix, _) in zip(envs, MODELS):
            bits = configure_env(env, tid, mix)
            obs, info = env.reset(seed=reset_seed)
            obs_list.append(obs)
            cmd_vels.append(info["cmd"]["x_vel"])
            cmd_tilts.append(info["cmd"]["tilt"])
            active_bits.append(bits)
            done_list.append(False)
        reset_seed += 1

    def render():
        for i, (env, spec) in enumerate(zip(envs, MODELS)):
            label = spec[0]
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
                task_id = (task_id + _sim_task_delta) % N_TASKS
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
                obs_s = StudentModel.obs(
                    obs_list[i], task_id, cmd_vels[i], cmd_tilts[i], active_bits[i]
                )
                action = model(torch.tensor(obs_s, dtype=torch.float32))
                obs, _, term, trunc, info = env.step(action)
                obs_list[i] = obs
                cmd_vels[i] = info["cmd"]["x_vel"]
                cmd_tilts[i] = info["cmd"]["tilt"]
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
