import os
from gymnasium import make
import numpy as np
import pygame
from stable_baselines3 import PPO

from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from utils.paths import MODELS_DIR
from wrappers.bipedal_walker.distill_env import DistillEnv

# =========================================

EXPERIMENT_NAME = "experts"
# MODEL_CHECKPOINT = "walk_forward"
MODEL_CHECKPOINT = "walk_backward"

# =========================================

_sim_paused = False
_sim_step = False
_sim_res = False


def main():
    global _sim_paused, _sim_step, _sim_res

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
            1: "walk back",
            2: "walk foward",
            3: "hop back",
            4: "hop back"
        }
    )
    
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    clock = pygame.time.Clock()
    
    # configure env and reset
    env.set_task(1);
    env.config_cmd_vel((-5.0, 0.0), 5, 0.5)
    env.config_hull_reset(x_range=(40, 80), y_range=(0.0, 0.5), vel_x_range=(-0.5, 0.5))
    obs, info = env.reset()
    cmd_x_vel = info["cmd"]["x_vel"]

    # wrap_env.action_space.seed(SEED)

    # load model
    print(f'Loading model "{MODEL_CHECKPOINT}"...')
    model_path = MODELS_DIR / f"{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}.zip"
    model = PPO.load(model_path, env=None, device="cpu")

    # manually render
    def render():
        frame = env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))  # type: ignore
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        clock.tick(50)

    while 1:
        pygame.event.pump()  # keep window alive on pause

        if _sim_res:
            # randomly choose a task
            env.set_task(np.random.choice([1, 2, 3, 4]))
            
            _sim_res = False
            obs, _ = env.reset()
            render()
            continue

        if _sim_paused:
            if not _sim_step:
                continue
            else:
                _sim_step = False

        assert env.action_space.shape is not None

        # append command to model input
        obs = np.append(obs, cmd_x_vel)
        
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        cmd_x_vel = info["cmd"]["x_vel"]  # update command

        render()

        if term or trunc:
            _sim_res = True


def on_press(key: Key | KeyCode | None) -> None:
    global _sim_paused, _sim_step, _sim_res

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
    elif k == "q":
        print("Exiting...")
        os._exit(0)


if __name__ == "__main__":
    main()
