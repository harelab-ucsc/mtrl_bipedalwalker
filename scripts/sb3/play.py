import os
from gymnasium import make
import pygame
from stable_baselines3 import PPO

from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from utils.paths import MODELS_DIR
from wrappers.bipedal_walker.standing_env import StandReward
from wrappers.plot_env import Plotter

# =========================================

EXPERIMENT_NAME = "stand_4-12_37_38-2026_04_01"
MODEL_CHECKPOINT = "best/best_model"
DRAW_PLOTS = False

# =========================================

_sim_paused = False
_sim_step = False
_sim_res = False


def main():
    global _sim_paused, _sim_step, _sim_res
    
    # start key listeners
    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # start to listen on a separate thread
    
    print(f"=== Starting experiment \"{EXPERIMENT_NAME}\" ===")
    
    # load env
    print("Loading environments...")
    env = make("BipedalWalker-v3", render_mode="human")
    
    wrap_env = StandReward(env)
    if DRAW_PLOTS:
        wrap_env = Plotter(wrap_env)
    
    obs, _ = wrap_env.reset()
    
    # wrap_env.action_space.seed(SEED)
    
    # load model
    print(f"Loading model \"{MODEL_CHECKPOINT}\"...")
    model_path = MODELS_DIR / f"{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}.zip"
    model = PPO.load(model_path, env=wrap_env)
    
    total_rewards = 0
    
    while(1):
        pygame.event.pump()  # keep window alive on pause
        
        if _sim_res:
            # print out total rewards before resetting
            print(f"Total rewards: {total_rewards}")
            total_rewards = 0
            
            _sim_res = False
            obs, _ = wrap_env.reset()
            continue
        
        if _sim_paused:
            if not _sim_step:
                continue
            else:
                _sim_step = False

        assert wrap_env.action_space.shape is not None
        
        action, _states = model.predict(obs)        
        obs, rewards, term, trunc, _ = wrap_env.step(action)
        total_rewards += float(rewards)
        
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
    
    if k == 'space':
        _sim_paused = not _sim_paused
        print("Paused" if _sim_paused else "Resumed")
    elif k == 's':
        _sim_step = True
    elif k == 'r':
        _sim_res = True
    elif k == 'q':
        print("Exiting...")
        os._exit(0)


if __name__ == "__main__":
    main()