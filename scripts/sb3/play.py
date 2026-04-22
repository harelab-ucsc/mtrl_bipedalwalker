import os
from gymnasium import make
import pygame
from stable_baselines3 import PPO

from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from utils.paths import MODELS_DIR
from wrappers.plot_env import Plotter
from wrappers.plot_reward_env import RewardPlotter
from wrappers.bipedal_walker.hop_env import HopEnv
from wrappers.bipedal_walker.hop_finetune_env import HopFTEnv
from wrappers.bipedal_walker.walk_env import WalkEnv
from wrappers.bipedal_walker.walk_finetune_env import WalkFTEnv
from wrappers.bipedal_walker.sit_env import SitEnv
from wrappers.bipedal_walker.proprio_wrapper import ProprioObsWrapper


# =========================================

# EXPERIMENT_NAME = "stand_8-18_50_45-2026_04_01"

# EXPERIMENT_NAME = "hop_forward/hop_forward_7-15_49_36-2026_04_09"
# EXPERIMENT_NAME = "hop_forward/hop_forward_7-17_00_23-2026_04_09"
# EXPERIMENT_NAME = "hop_forward/hop_forward_8-01_47_33-2026_04_16"
# EXPERIMENT_NAME = "hop_forward/hop_forward_8_1-01_50_21-2026_04_16"

# EXPERIMENT_NAME = "hop_backward/hop_backward_2-20_35_48-2026_04_09"
# EXPERIMENT_NAME = "hop_backward/hop_backward_3_2-15_46_14-2026_04_15"

# EXPERIMENT_NAME = "walk_forward/walk_forward_9-00_50_10-2026_04_12"
# EXPERIMENT_NAME = "walk_forward/walk_forward_10-15_47_52-2026_04_12"
# EXPERIMENT_NAME = "walk_forward/walk_forward_11-02_05_14-2026_04_16"
# EXPERIMENT_NAME = "walk_forward/walk_forward_11_1-03_09_34-2026_04_16"
EXPERIMENT_NAME = "walk_forward/walk_forward_11-02_05_14-2026_04_16"

# EXPERIMENT_NAME = "walk_backward/walk_backward_7_1"
# EXPERIMENT_NAME = "walk_backward/walk_backward_7_1"
# EXPERIMENT_NAME = "walk_backward/walk_backward_7_1"
# EXPERIMENT_NAME = "walk_backward/walk_backward_7_2-00_06_27-2026_04_15"
# EXPERIMENT_NAME = "walk_backward/walk_backward_7_3-00_06_54-2026_04_15"
# EXPERIMENT_NAME = "walk_backward/walk_backward_7_4-00_07_13-2026_04_15"
# EXPERIMENT_NAME = "walk_backward/walk_backward_7_5-00_16_04-2026_04_15"
# EXPERIMENT_NAME = "walk_backward/walk_backward_7_6-00_22_45-2026_04_15"
MODEL_CHECKPOINT = "best/best_model"
# None  → no plots
# "obs" → proprioceptive observation dashboard (Plotter)
# "reward" → per-term reward breakdown dashboard (RewardPlotter)
PLOT_MODE: str | None = None

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

    wrap_env = ProprioObsWrapper(
        WalkEnv(
            env,
            ep_time=15,
            vel_switching_freq=3,
            vel_sample_range=(-5, 0),
            hull_x_range=(40, 80),
            vel_sample_zero=0.5,
            vel_interp_speed=0.5,
        )
    )
    # wrap_env = env
    if PLOT_MODE == "obs":
        wrap_env = Plotter(wrap_env)
    elif PLOT_MODE == "reward":
        wrap_env = RewardPlotter(wrap_env)

    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    clock = pygame.time.Clock()

    obs, _ = wrap_env.reset()

    # wrap_env.action_space.seed(SEED)

    # load model
    print(f'Loading model "{MODEL_CHECKPOINT}"...')
    model_path = MODELS_DIR / f"{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}.zip"
    model = PPO.load(model_path, env=wrap_env, device="cpu")

    total_rewards = 0

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

        action, _states = model.predict(obs, deterministic=True)
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
