import pygame
import numpy as np
from gymnasium import make
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import os

from wrappers.plot_env import Plotter
from wrappers.plot_reward_env import RewardPlotter
from wrappers.bipedal_walker.hopping_env import HopReward
from wrappers.bipedal_walker.walking_env import WalkReward
from wrappers.bipedal_walker.proprio_wrapper import ProprioObsWrapper

SEED = 42
# None     → no plots
# "obs"    → proprioceptive observation dashboard (Plotter)
# "reward" → per-term reward breakdown dashboard (RewardPlotter)
PLOT_MODE: str | None = None

_sim_paused = False
_sim_step = False
_sim_res = False


def main():
    global _sim_paused, _sim_step, _sim_res

    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # start to listen on a separate thread

    print("Loading environments...")

    env = make(
        "BipedalWalker-v3", render_mode="rgb_array"
    )  # no autodisplay, we'll use our own wrapper render
    
    wrap_env = ProprioObsWrapper(    
        WalkReward(
            env,
            ep_time=15,
            vel_switching_freq=3,
            vel_sample_range=(0, 5),
            vel_sample_zero=0.05,
        )
    )
    if PLOT_MODE == "obs":
        wrap_env = Plotter(wrap_env)
    elif PLOT_MODE == "reward":
        wrap_env = RewardPlotter(wrap_env)

    obs = wrap_env.observation_space
    act = wrap_env.action_space
    
    print(f"obs  {obs.shape}  {obs.dtype}")
    print(f"act  {act.shape}  [{act.low[0]:.1f}, {act.high[0]:.1f}]") # type: ignore

    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    clock = pygame.time.Clock()

    wrap_env.reset(seed=SEED)
    wrap_env.action_space.seed(SEED)

    # manually render
    def render():
        frame = wrap_env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))  # type: ignore
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        clock.tick(50)

    while 1:
        pygame.event.pump()  # keep window alive on pause

        if _sim_res:
            _sim_res = False
            wrap_env.reset()
            render()
            continue

        if _sim_paused:
            if not _sim_step:
                continue
            else:
                _sim_step = False

        assert wrap_env.action_space.shape is not None

        # random agent
        action = wrap_env.action_space.sample()

        # zero agent
        # action = np.zeros(wrap_env.action_space.shape)

        _, _, term, trunc, _ = wrap_env.step(action)
        print("=== Testing with action: ", action)

        render()

        if term or trunc:
            _sim_res = True
        # print()
        # print();


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
