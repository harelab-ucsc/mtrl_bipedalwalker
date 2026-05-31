import os
from gymnasium import make
import pygame
from stable_baselines3 import PPO

from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from utils.paths import MODELS_DIR
from wrappers.plot_env import Plotter
from wrappers.plot_reward_env import RewardPlotter
from wrappers.ppo_bc.ppo_bc_env import RlFTEnv


# =========================================

# EXPERIMENT_NAME = "ppo_bc/1.0.0-18_57_32-2026_05_24"
# EXPERIMENT_NAME = "ppo_bc/1.0.1-17_30_08-2026_05_25"
# EXPERIMENT_NAME = "ppo_bc/1.0.2-23_53_29-2026_05_25"
# EXPERIMENT_NAME = "ppo_bc/1.0.3-11_13_43-2026_05_26"
# EXPERIMENT_NAME = "ppo_bc/1.1.0-16_07_40-2026_05_26"
# EXPERIMENT_NAME = "ppo_bc/1.2.0-21_04_31-2026_05_26"
# EXPERIMENT_NAME = "ppo_bc/critic_pretrain/1.0.2.1-15_24_49-2026_05_27"
# EXPERIMENT_NAME = "ppo_bc/1.2.1-01_01_25-2026_05_27"
EXPERIMENT_NAME = "ppo_bc_adv/pretrain/1.0.1"
MODEL_CHECKPOINT = "rl_model_2799664_steps"
# None  → no plots
# "obs" → proprioceptive observation dashboard (Plotter)
# "reward" → per-term reward breakdown dashboard (RewardPlotter)
PLOT_MODE: str | None = None

MANUAL_CTRL = True

# --- env params (mirrors scripts/ppo_bc/train.py defaults) ---
EP_TIME              = 10
CMD_SWITCHING_TIME   = (3.0, 4.0)   # (vel, tilt)
TASK_SWITCHING_TIME  = 6.0
CMD_INTERP_SPEED     = (5.0, 1.0)
CMD_SAMPLE_RANGE     = ((-5.0, 5.0), (-0.75, 0.75))
CMD_SAMPLE_ZERO      = (0.2, 0.15)
ALLOWED_TASK_MIXING  = [
    (1, 0, 0),  # walk
    (0, 1, 0),  # flamingo
    (0, 0, 1),  # tilt
]
HULL_X_RANGE         = (20.0, 60.0)

FPS = 50
VEL_KEY_SPEED  = 5.0    # m/s per second; rate of vel target accumulation
TILT_KEY_SPEED = 1.0    # rad/s; rate of tilt target accumulation

# =========================================

_sim_paused = False
_sim_step = False
_sim_res = False
_left_held = False
_right_held = False
_up_held = False
_down_held = False
_zero_cmds = False
_task_set: tuple[int, int, int] | None = None  # None = no change


def main():
    global _sim_paused, _sim_step, _sim_res
    global _left_held, _right_held, _up_held, _down_held, _zero_cmds, _task_set

    # start key listeners
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()  # start to listen on a separate thread

    print(f'=== Starting experiment "{EXPERIMENT_NAME}" ===')
    if MANUAL_CTRL:
        print("Controls:")
        print("  r           reset")
        print("  space       pause / resume")
        print("  s           step (while paused)")
        print("  q           quit")
        print("  1-5         toggle tasks")
        print("  left/right  velocity -/+")
        print("  up/down     tilt +/-")
        print("  0           zero cmds")

    # load env
    print("Loading environments...")
    raw = make("BipedalWalker-v3", render_mode="rgb_array")

    rlft_env = RlFTEnv(
        raw,
        ep_time=EP_TIME,
        cmd_switching_time=CMD_SWITCHING_TIME,
        task_switching_time=TASK_SWITCHING_TIME,
        cmd_interp_speed=CMD_INTERP_SPEED,
        cmd_sample_range=CMD_SAMPLE_RANGE,
        cmd_sample_zero=CMD_SAMPLE_ZERO,
        allowed_task_mixing=ALLOWED_TASK_MIXING,
        hull_x_range=HULL_X_RANGE,
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
    cmd_tilt_target = 0.0
    task_vec: tuple[int, int, int] = ALLOWED_TASK_MIXING[0]  # default walk
    total_rewards = 0

    def do_reset():
        obs, _ = wrap_env.reset()
        if MANUAL_CTRL:
            cmd = (cmd_vel_target, cmd_tilt_target)
            rlft_env._cmd_vec = cmd
            rlft_env._cmd_vec_target = cmd
            rlft_env._task_id_vec = task_vec
            # rebuild obs with the just-applied cmd / task (strip last 5 = cmd(2)+task(3))
            base = obs[:-5]
            obs = rlft_env._derive_full_obs(
                base, rlft_env._effective_cmd_vec(), task_vec
            )
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
                cmd_vel_target = min(
                    cmd_vel_target + VEL_KEY_SPEED / FPS, CMD_SAMPLE_RANGE[0][1]
                )
            if _left_held:
                cmd_vel_target = max(
                    cmd_vel_target - VEL_KEY_SPEED / FPS, CMD_SAMPLE_RANGE[0][0]
                )
            if _up_held:
                cmd_tilt_target = min(
                    cmd_tilt_target + TILT_KEY_SPEED / FPS, CMD_SAMPLE_RANGE[1][1]
                )
            if _down_held:
                cmd_tilt_target = max(
                    cmd_tilt_target - TILT_KEY_SPEED / FPS, CMD_SAMPLE_RANGE[1][0]
                )
            if _zero_cmds:
                cmd_vel_target = 0.0
                cmd_tilt_target = 0.0
                _zero_cmds = False
            rlft_env._cmd_vec_target = (cmd_vel_target, cmd_tilt_target)

            if _task_set is not None:
                task_vec = _task_set
                _task_set = None
                rlft_env._task_id_vec = task_vec

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
    global _left_held, _right_held, _up_held, _down_held, _zero_cmds, _task_set

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
    elif k == "up":
        _up_held = True
    elif k == "down":
        _down_held = True
    elif k == "0":
        _zero_cmds = True
    elif k == "1":
        _task_set = (1, 0, 0)
        print("Task: walk")
    elif k == "2":
        _task_set = (0, 1, 0)
        print("Task: flamingo")
    elif k == "3":
        _task_set = (0, 0, 1)
        print("Task: tilt")
    elif k == "4":
        _task_set = (1, 1, 0)
        print("Task: walk + flamingo")
    elif k == "5":
        _task_set = (1, 0, 1)
        print("Task: walk + tilt")
    elif k == "q":
        print("Exiting...")
        os._exit(0)


def on_release(key: Key | KeyCode | None) -> None:
    global _left_held, _right_held, _up_held, _down_held

    if isinstance(key, Key):
        if key.name == "left":
            _left_held = False
        elif key.name == "right":
            _right_held = False
        elif key.name == "up":
            _up_held = False
        elif key.name == "down":
            _down_held = False


if __name__ == "__main__":
    main()
