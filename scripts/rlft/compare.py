import os
import warnings
from gymnasium import make
import numpy as np
import pygame
from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from stable_baselines3 import PPO
from utils.paths import MODELS_DIR
from wrappers.bipedal_walker.rltf_env import RlFTEnv

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================================

EXPERIMENTS = [
    ("rlft/finetuned/ml_3.3.1_g97-15_16_22-2026_05_11", "3.3.1"),
    ("rlft/finetuned/ml_3.3.1a_g97-15_16_31-2026_05_11", "3.3.1a"),
]

MODEL_CHECKPOINT = "best/best_model"

ENV_W, ENV_H = 600, 400
MAX_COLS = 2
FPS = 50

# Rate at which velocity target changes when arrow key is held (m/s per second).
# Also controls the interpolation speed of _cmd_vel toward the target.
VEL_KEY_SPEED = 10.0

# =========================================

_sim_paused = False
_sim_step = False
_sim_res = False
_left_held = False
_right_held = False
_vel_to_zero = False
_task_set = -1  # -1 = no change, 0 = walk, 1 = hop


def main():
    global _sim_paused, _sim_step, _sim_res
    global _left_held, _right_held, _vel_to_zero, _task_set

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print(f'=== Comparison: {[label for _, label in EXPERIMENTS]} ===')
    print("Controls: r=reset, w=walk, h=hop, left/right=velocity, down=stop, space=pause, s=step, q=quit")

    print("Loading environments...")
    envs: list[RlFTEnv] = []
    for _ in EXPERIMENTS:
        raw = make("BipedalWalker-v3", render_mode="rgb_array")
        env = RlFTEnv(
            raw,
            manual_ctrl_mode=True,
            hull_x_range=(0, 0),
            vel_interp_speed=VEL_KEY_SPEED,
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
    task_id = 0

    def reset_all():
        nonlocal reset_seed
        obs_list.clear()
        done_list.clear()
        for env in envs:
            obs, _ = env.reset(seed=reset_seed)
            env._cmd_vel = cmd_vel_target
            env._cmd_vel_target = cmd_vel_target
            env._cmd_task_id = task_id
            base_obs = obs[:-3]
            obs = env._derive_full_obs(base_obs, cmd_vel_target, task_id)
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

        # drive cmd_vel_target while arrow keys are held
        if _right_held:
            cmd_vel_target = min(cmd_vel_target + VEL_KEY_SPEED / FPS, 5.0)
        if _left_held:
            cmd_vel_target = max(cmd_vel_target - VEL_KEY_SPEED / FPS, -5.0)
        if _vel_to_zero:
            cmd_vel_target = 0.0
            _vel_to_zero = False

        # push updated target to all envs each frame
        for env in envs:
            env._cmd_vel_target = cmd_vel_target

        if _task_set != -1:
            task_id = _task_set
            _task_set = -1
            for env in envs:
                env._cmd_task_id = task_id

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
    global _left_held, _right_held, _vel_to_zero, _task_set

    if isinstance(key, KeyCode):
        k = key.char
    elif isinstance(key, Key):
        k = key.name
    else:
        return

    if k == "left":
        _left_held = True
    elif k == "right":
        _right_held = True
    elif k == "down":
        _vel_to_zero = True
    elif k == "space":
        _sim_paused = not _sim_paused
        print("Paused" if _sim_paused else "Resumed")
    elif k == "s":
        _sim_step = True
    elif k == "r":
        _sim_res = True
    elif k == "w":
        _task_set = 0
        print("Task: walk")
    elif k == "h":
        _task_set = 1
        print("Task: hop")
    elif k == "q":
        print("Exiting...")
        os._exit(0)


def on_release(key: Key | KeyCode | None) -> None:
    global _left_held, _right_held

    if isinstance(key, Key):
        if key.name == "left":
            _left_held = False
        elif key.name == "right":
            _right_held = False


if __name__ == "__main__":
    main()
