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
from wrappers.bipedal_walker.rltf_env import RlFTEnv
from wrappers.bipedal_walker.proprio_wrapper import ProprioObsWrapper


# =========================================

# EXPERIMENT_NAME = "rlft/finetuned/ml_1-16_09_36-2026_05_05"
# EXPERIMENT_NAME = "rlft/finetuned/ml_2_g95-02_20_25-2026_05_07"
# EXPERIMENT_NAME = "rlft/finetuned/ml_3.1_g99-17_38_43-2026_05_07"
# EXPERIMENT_NAME = "rlft/finetuned/ml_3.2_g99-17_57_41-2026_05_07"
# EXPERIMENT_NAME = "rlft/finetuned/ml_3.2.1_g99-18_32_27-2026_05_07"
# EXPERIMENT_NAME = "rlft/finetuned/ml_3.2.2_g99-20_41_51-2026_05_07"
# EXPERIMENT_NAME = "rlft/finetuned/ml_3.2.3_g95-13_26_23-2026_05_08"
# EXPERIMENT_NAME = "rlft/finetuned/ml_3.2.4_g95-13_48_36-2026_05_08"
# EXPERIMENT_NAME = "rlft/finetuned/ml_3.2.5_g97-15_00_01-2026_05_08"
# EXPERIMENT_NAME = "rlft/finetuned/ml_3.2.6_g97-15_21_59-2026_05_08"
# EXPERIMENT_NAME = "rlft/finetuned/ml_3.2.7_g97-15_52_42-2026_05_08"
# EXPERIMENT_NAME = "rlft/finetuned/ml_3.2.8_g97-19_04_21-2026_05_08"
# EXPERIMENT_NAME = "rlft/finetuned/xl_3.2.8a_g97-19_04_25-2026_05_08"
# EXPERIMENT_NAME = "rlft/finetuned/ml_3.3.1_g97-15_16_22-2026_05_11"
# EXPERIMENT_NAME = "rlft/finetuned/ml_3.3.1a_g97-15_16_31-2026_05_11"
# EXPERIMENT_NAME = "rlft/finetuned/ml_3.3.2_g97-01_08_13-2026_05_12"
EXPERIMENT_NAME = "rlft/finetuned/ml_3.3.2a_g97-01_08_38-2026_05_12"
MODEL_CHECKPOINT = "best/best_model"
# None  → no plots
# "obs" → proprioceptive observation dashboard (Plotter)
# "reward" → per-term reward breakdown dashboard (RewardPlotter)
PLOT_MODE: str | None = None

MANUAL_CTRL = True  # turn on for arrow key ctrl
FPS = 50
VEL_KEY_SPEED = 5.0  # m/s per second; rate of target change and interpolation speed

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

    # start key listeners
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()  # start to listen on a separate thread

    print(f'=== Starting experiment "{EXPERIMENT_NAME}" ===')
    if MANUAL_CTRL:
        print("Controls: r=reset, w=walk, h=hop, left/right=velocity, down=stop, space=pause, s=step, q=quit")

    # load env
    print("Loading environments...")
    raw = make("BipedalWalker-v3", render_mode="rgb_array")

    rlft_env = RlFTEnv(
        raw,
        vel_switching_freq=3,
        task_switching_freq=6,
        vel_interp_speed=VEL_KEY_SPEED if MANUAL_CTRL else 3.0,
        manual_ctrl_mode=MANUAL_CTRL,
    )
    wrap_env = rlft_env
    if PLOT_MODE == "obs":
        wrap_env = Plotter(rlft_env)
    elif PLOT_MODE == "reward":
        wrap_env = RewardPlotter(rlft_env)

    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    clock = pygame.time.Clock()

    # load model
    print(f'Loading model "{MODEL_CHECKPOINT}"...')
    model_path = MODELS_DIR / f"{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}.zip"
    model = PPO.load(model_path, env=wrap_env, device="cpu")

    cmd_vel_target = 0.0
    task_id = 0
    total_rewards = 0

    def do_reset():
        obs, _ = wrap_env.reset()
        if MANUAL_CTRL:
            rlft_env._cmd_vel = cmd_vel_target
            rlft_env._cmd_vel_target = cmd_vel_target
            rlft_env._cmd_task_id = task_id
            obs = rlft_env._derive_full_obs(obs[:-3], cmd_vel_target, task_id)
        return obs

    obs = do_reset()

    # manually render
    def render():
        frame = wrap_env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))  # type: ignore
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        clock.tick(FPS)

    while 1:
        pygame.event.pump()  # keep window alive on pause

        if MANUAL_CTRL:
            if _right_held:
                cmd_vel_target = min(cmd_vel_target + VEL_KEY_SPEED / FPS, 5.0)
            if _left_held:
                cmd_vel_target = max(cmd_vel_target - VEL_KEY_SPEED / FPS, -5.0)
            if _vel_to_zero:
                cmd_vel_target = 0.0
                _vel_to_zero = False
            rlft_env._cmd_vel_target = cmd_vel_target

            if _task_set != -1:
                task_id = _task_set
                _task_set = -1
                rlft_env._cmd_task_id = task_id

        if _sim_res:
            # print out total rewards before resetting
            print(f"Total rewards: {total_rewards}")
            total_rewards = 0

            _sim_res = False
            obs = do_reset()
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

        if term or (not MANUAL_CTRL and trunc):
            _sim_res = True


def on_press(key: Key | KeyCode | None) -> None:
    global _sim_paused, _sim_step, _sim_res
    global _left_held, _right_held, _vel_to_zero, _task_set

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
    elif k == "down":
        _vel_to_zero = True
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
