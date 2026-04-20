# https://arxiv.org/pdf/2505.11164

import os

from gymnasium import make
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
from utils.paths import MODELS_DIR, LOGS_DIR, ROOT
from wrappers.bipedal_walker.proprio_wrapper import ProprioObsWrapper

EXPERT_MODELS = [
    "experts/hop_backward",
    "experts/hop_forward",
    "experts/walk_backward",
    "experts/walk_forawrd"
]

_sim_paused = False
_sim_step = False
_sim_res = False


def main():
    global _sim_paused, _sim_step, _sim_res
    
    # start key listeners
    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # start to listen on a separate thread
    
    print("Loading environments...")
    
    env = make("BipedalWalker-v3", render_mode="rgb_array")
    env = ProprioObsWrapper(env)  # remove lidar
    
    # DAgger
    #
    
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