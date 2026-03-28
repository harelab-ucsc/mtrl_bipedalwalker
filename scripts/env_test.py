import numpy as np
from gymnasium import make
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import os

from wrappers.bipedal_walker.plot_env import Plot_Env

SEED = 42

_sim_paused = False
_sim_step = False


def main():
    global _sim_paused, _sim_step
    
    env = make("BipedalWalker-v3", render_mode="human")
    wrap_env = Plot_Env(env)
    wrap_env.reset(seed=SEED)
    wrap_env.action_space.seed(SEED)
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # start to listen on a separate thread
    
    while(1):
        if _sim_paused:
            if not _sim_step:
                continue
            else:
                _sim_step = False
        
        assert wrap_env.action_space.shape is not None
        
        # random agent
        action = wrap_env.action_space.sample()
        wrap_env.step(action)
        print("=== Testing with action: ", action)
        # print()
        # print();


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
