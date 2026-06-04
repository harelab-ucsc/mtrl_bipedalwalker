"""
scripts/rlft/compare.py
=======================

Render several finetuned Rudin-baseline models side-by-side in a grid, all
driven by the same keyboard input (v2 controls, mirrors scripts/rlft/play.py).
Envs share a seed each round so the comparison is apples-to-apples.

Controls (applied to all models):
  r           reset all
  space       pause / resume
  s           step (while paused)
  q           quit
  1-5         toggle tasks (walk / flamingo / tilt / walk+flamingo / walk+tilt)
  left/right  velocity -/+
  up/down     tilt +/-
  0           zero cmds
"""

import os
import warnings
from gymnasium import make
import numpy as np
import pygame
from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from stable_baselines3 import PPO
from utils.paths import MODELS_DIR
from wrappers.ppo_bc.ppo_bc_env import RlFTEnv
from mdp.bipedal_walker.tasks import GAIT, ONEHOT

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================================

# (experiment_name, label). experiment_name is a models/-relative dir, e.g.
# rudin/finetuned/1.0.0 or rudin_adv/finetuned/1.0.0.
EXPERIMENTS = [
    # ("rudin/finetuned/1.0.0", "rudin 1.0.0"),
    # ("rudin_adv/finetuned/1.0.0", "rudin_adv 1.0.0"),
]

MODEL_CHECKPOINT = "best/best_model"

ENV_W, ENV_H = 600, 400
MAX_COLS = 2
FPS = 50

# obs-bit scheme: GAIT (default, 2.x.x) or ONEHOT (legacy 1.x.x). Picks the keyboard
# task menu + how the 3 obs bits are interpreted by the loaded models.
TASK_SCHEME = GAIT

# --- env params (mirror finetune_config.py) ---
EP_TIME              = 10
CMD_INTERP_SPEED     = (5.0, 1.0)
CMD_SAMPLE_RANGE     = ((-5.0, 5.0), (-0.75, 0.75))
HULL_X_RANGE         = (20.0, 60.0)

VEL_KEY_SPEED  = 5.0    # m/s per second
TILT_KEY_SPEED = 1.0    # rad/s
DEFAULT_VEL    = 3.0    # m/s default vel a gait task key dials in
DEFAULT_TILT   = 0.5    # rad default tilt a tilt task key dials in

# Keyboard task menus: key -> (label, 3-bit task vector, vel, tilt). vel/tilt are
# auto-dialed targets (None leaves the arrow-key target untouched).
GAIT_TASK_MENU: dict[str, tuple] = {
    "1": ("walk forward",  (1, 0, 0),  DEFAULT_VEL, 0.0),
    "2": ("walk backward", (1, 0, 0), -DEFAULT_VEL, 0.0),
    "3": ("hop forward",   (0, 1, 0),  DEFAULT_VEL, 0.0),
    "4": ("hop backward",  (0, 1, 0), -DEFAULT_VEL, 0.0),
    "5": ("tilt",          (1, 0, 0),  0.0,         DEFAULT_TILT),
    "6": ("walk + tilt",   (1, 0, 0),  DEFAULT_VEL, DEFAULT_TILT),
}
ONEHOT_TASK_MENU: dict[str, tuple] = {
    "1": ("walk",            (1, 0, 0), None, None),
    "2": ("flamingo",        (0, 1, 0), None, None),
    "3": ("tilt",            (0, 0, 1), None, None),
    "4": ("walk + flamingo", (1, 1, 0), None, None),
    "5": ("walk + tilt",     (1, 0, 1), None, None),
}
TASK_MENU = GAIT_TASK_MENU if TASK_SCHEME == GAIT else ONEHOT_TASK_MENU
DEFAULT_TASK_VEC: tuple[int, int, int] = (1, 0, 0)

# =========================================

_sim_paused = False
_sim_step = False
_sim_res = False
_left_held = False
_right_held = False
_up_held = False
_down_held = False
_zero_cmds = False
# pending menu selection: (task_bits, vel|None, tilt|None) or None for no change.
_task_set: tuple[tuple[int, int, int], float | None, float | None] | None = None


def main():
    global _sim_paused, _sim_step, _sim_res
    global _left_held, _right_held, _up_held, _down_held, _zero_cmds, _task_set

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print(f"=== Comparison: {[label for _, label in EXPERIMENTS]} ===")
    print("Controls: r=reset, 1-5=tasks, left/right=vel, up/down=tilt, 0=zero, space=pause, s=step, q=quit")

    print("Loading environments...")
    envs: list[RlFTEnv] = []
    for _ in EXPERIMENTS:
        raw = make("BipedalWalker-v3", render_mode="rgb_array")
        env = RlFTEnv(
            raw,
            ep_time=EP_TIME,
            cmd_interp_speed=CMD_INTERP_SPEED,
            cmd_sample_range=CMD_SAMPLE_RANGE,
            hull_x_range=HULL_X_RANGE,
            manual_ctrl_mode=True,
            task_scheme=TASK_SCHEME,
        )
        envs.append(env)

    print("Loading models...")
    models: list[PPO] = []
    for i, (exp_name, label) in enumerate(EXPERIMENTS):
        model_path = MODELS_DIR / f"{exp_name}/{MODEL_CHECKPOINT}.zip"
        m = PPO.load(model_path, env=envs[i], device="cpu")
        models.append(m)
        print(f'  Loaded "{label}"')

    pygame.init()
    pygame.font.init()
    n_cols = min(len(envs), MAX_COLS)
    n_rows = -(-len(envs) // MAX_COLS)
    screen = pygame.display.set_mode((ENV_W * n_cols, ENV_H * n_rows))
    clock = pygame.time.Clock()
    label_font = pygame.font.SysFont("Courier New", 32, bold=True)

    obs_list: list[np.ndarray] = []
    done_list: list[bool] = []
    reset_seed = 420
    cmd_vel_target = 0.0
    cmd_tilt_target = 0.0
    task_vec: tuple[int, int, int] = (1, 0, 0)  # default walk

    def reset_all():
        nonlocal reset_seed
        obs_list.clear()
        done_list.clear()
        for env in envs:
            obs, _ = env.reset(seed=reset_seed)
            cmd = (cmd_vel_target, cmd_tilt_target)
            env._cmd_vec = cmd
            env._cmd_vec_target = cmd
            env._task_id_vec = task_vec
            obs = env._derive_full_obs(obs[:-5], env._effective_cmd_vec(), task_vec)
            obs_list.append(obs)
            done_list.append(False)
        reset_seed += 1

    def render():
        for i, (env, (_, label)) in enumerate(zip(envs, EXPERIMENTS)):
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
        clock.tick(FPS)

    reset_all()

    while True:
        pygame.event.pump()

        # drive cmd targets while arrow keys are held
        if _right_held:
            cmd_vel_target = min(cmd_vel_target + VEL_KEY_SPEED / FPS, CMD_SAMPLE_RANGE[0][1])
        if _left_held:
            cmd_vel_target = max(cmd_vel_target - VEL_KEY_SPEED / FPS, CMD_SAMPLE_RANGE[0][0])
        if _up_held:
            cmd_tilt_target = min(cmd_tilt_target + TILT_KEY_SPEED / FPS, CMD_SAMPLE_RANGE[1][1])
        if _down_held:
            cmd_tilt_target = max(cmd_tilt_target - TILT_KEY_SPEED / FPS, CMD_SAMPLE_RANGE[1][0])
        if _zero_cmds:
            cmd_vel_target = 0.0
            cmd_tilt_target = 0.0
            _zero_cmds = False

        # push updated target to all envs each frame
        for env in envs:
            env._cmd_vec_target = (cmd_vel_target, cmd_tilt_target)

        if _task_set is not None:
            task_vec, _set_vel, _set_tilt = _task_set
            _task_set = None
            for env in envs:
                env._task_id_vec = task_vec
            # auto-dial the task's preset commands (gait menu); None leaves the
            # arrow-key-driven target untouched (onehot menu).
            if _set_vel is not None:
                cmd_vel_target = _set_vel
            if _set_tilt is not None:
                cmd_tilt_target = _set_tilt

        if _sim_res:
            reset_all()
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
            action, _ = model.predict(obs_list[i], deterministic=True)
            obs, _, term, _, _ = env.step(action)
            obs_list[i] = obs
            if term:
                done_list[i] = True

        render()
        # no auto-reset when all done — press r to restart


def on_press(key: Key | KeyCode | None) -> None:
    global _sim_paused, _sim_step, _sim_res
    global _left_held, _right_held, _up_held, _down_held, _zero_cmds, _task_set

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
    elif k == "left":
        _left_held = True
    elif k == "right":
        _right_held = True
    elif k == "up":
        _up_held = True
    elif k == "down":
        _down_held = True
    elif k == "0":
        _zero_cmds = True
    elif k in TASK_MENU:
        label, bits, vel, tilt = TASK_MENU[k]
        _task_set = (bits, vel, tilt)
        print(f"Task: {label}")
    elif k == "q":
        print("Exiting...")
        os._exit(0)


def on_release(key: Key | KeyCode | None) -> None:
    global _left_held, _right_held, _up_held, _down_held

    if isinstance(key, Key):
        if key.name == "left":
            _left_held = False
        elif key.name == "right":
            _right_held = False
        elif key.name == "up":
            _up_held = False
        elif key.name == "down":
            _down_held = False


if __name__ == "__main__":
    main()
