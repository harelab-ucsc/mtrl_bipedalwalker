import os
import gymnasium as gym
from stable_baselines3 import PPO

from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from utils.paths import MODELS_DIR
from wrappers.bipedal_walker.plot_env import Plot_Env

# =========================================

EXPERIMENT_NAME = "3_25_2026/basic_walker_5-00_39_40-2026_03_26"
MODEL_CHECKPOINT = "best/best_model"

# =========================================

_sim_paused = False
_sim_step = False

env = gym.make("BipedalWalker-v3", render_mode="human")
wrap_env = Plot_Env(env)
wrap_env.reset()

model_path = MODELS_DIR / f"{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}.zip"
model = PPO.load(model_path, env=wrap_env)

episodes = 5

def main():
    global _sim_paused, _sim_step
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # start to listen on a separate thread
    
    for ep in range(episodes):
        obs, _ = wrap_env.reset()
        done = False
        total_rewards = 0
        
        while not done:
            if _sim_paused:
                if not _sim_step:
                    continue
                else:
                    _sim_step = False
                
            action, _states = model.predict(obs)
            obs, rewards, done, info, _ = wrap_env.step(action)
            total_rewards += float(rewards)
            wrap_env.render()
        print(total_rewards)


def on_press(key: Key | KeyCode | None) -> None:
    global _sim_paused, _sim_step
    
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
    elif k == 'q':
        print("Exiting...")
        os._exit(0)


if __name__ == "__main__":
    main()