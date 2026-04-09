import os
from gymnasium import make
import pygame
from stable_baselines3 import PPO

from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from utils.paths import MODELS_DIR
from wrappers.bipedal_walker.standing_env import StandReward
from wrappers.bipedal_walker.hopping_env import HopReward
from wrappers.bipedal_walker.hopping_env_proprio import ProprioHopReward
from wrappers.plot_env import Plotter


# =========================================

# EXPERIMENT_NAME = "stand_8-18_50_45-2026_04_01"
EXPERIMENT_NAME = "hop_forward/hop_forward_6-13_45_02-2026_04_09"
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
    env = make("BipedalWalker-v3", render_mode="rgb_array")
    
    # wrap_env = StandReward(env, disturbance_freq=3, disturbance_force=((-3, 5), (0, 1)))
    wrap_env = ProprioHopReward(
        env,
        ep_time=15,
        vel_switching_freq=3,
        vel_sample_range=(-5, 0),
        vel_sample_zero=0.05,
    )
    # wrap_env = env
    if DRAW_PLOTS:
        wrap_env = Plotter(wrap_env)
    
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    clock = pygame.time.Clock()
    
    obs, _ = wrap_env.reset()
    
    # wrap_env.action_space.seed(SEED)
    
    # load model
    print(f"Loading model \"{MODEL_CHECKPOINT}\"...")
    model_path = MODELS_DIR / f"{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}.zip"
    model = PPO.load(model_path, env=wrap_env)
    
    total_rewards = 0
    
    # manually render
    def render():
        frame = wrap_env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))  # type: ignore
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        clock.tick(50)
    
    while(1):
        pygame.event.pump()  # keep window alive on pause
        
        if _sim_res:
            # print out total rewards before resetting
            print(f"Total rewards: {total_rewards}")
            total_rewards = 0
            
            _sim_res = False
            obs, _ = wrap_env.reset()
            render()
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