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
from mdp.bipedal_walker.tasks import GAIT, ONEHOT, SINGLE_TASKS_GAIT


# =========================================

EXPERIMENT_NAME = "ppo_bc_adv/pretrain/3.0.0"
MODEL_CHECKPOINT = "final"
# None  → no plots
# "obs" → proprioceptive observation dashboard (Plotter)
# "reward" → per-term reward breakdown dashboard (RewardPlotter)
PLOT_MODE: str | None = None

MANUAL_CTRL = True

# obs-bit scheme: GAIT (default, 2.x.x) or ONEHOT (legacy 1.x.x). Drives the
# RlFTEnv task_scheme, allowed_task_mixing, and the keyboard task menu.
TASK_SCHEME = GAIT

# --- env params (mirrors scripts/ppo_bc/train.py defaults) ---
EP_TIME              = 10
CMD_SWITCHING_TIME   = (3.0, 4.0)   # (vel, tilt)
TASK_SWITCHING_TIME  = 6.0
CMD_INTERP_SPEED     = (5.0, 1.0)
CMD_SAMPLE_RANGE     = ((-5.0, 5.0), (-0.75, 0.75))
CMD_SAMPLE_ZERO      = (0.2, 0.15)
# Gait: the canonical GaitTask single tasks (gait + command ranges). Onehot:
# legacy per-task one-hot rows. ALLOWED_TASK_MIXING is what the env samples from
# (manual mode pins the task instead, but the env still needs a valid set).
ALLOWED_TASK_MIXING_ONEHOT = [
    (1, 0, 0),  # walk
    (0, 1, 0),  # flamingo
    (0, 0, 1),  # tilt
]
ALLOWED_TASK_MIXING = (
    list(SINGLE_TASKS_GAIT) if TASK_SCHEME == GAIT else ALLOWED_TASK_MIXING_ONEHOT
)
# default 3-bit task vector + commands the keyboard menu / reset start from.
DEFAULT_TASK_VEC: tuple[int, int, int] = (1, 0, 0)  # two-leg walk / onehot walk
HULL_X_RANGE         = (20.0, 60.0)

FPS = 50
VEL_KEY_SPEED  = 5.0    # m/s per second; rate of vel target accumulation
TILT_KEY_SPEED = 1.0    # rad/s; rate of tilt target accumulation
DEFAULT_VEL    = 3.0    # m/s default vel a gait task key dials in (walk/hop fwd)
DEFAULT_TILT   = 0.5    # rad default tilt a tilt task key dials in

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
# default (vel, tilt) a task key dials in alongside the bits (gait menu); None =
# leave the current command targets untouched (onehot menu / no task key).
_cmd_set: tuple[float, float] | None = None


def main():
    global _sim_paused, _sim_step, _sim_res
    global _left_held, _right_held, _up_held, _down_held, _zero_cmds, _task_set, _cmd_set

    # start key listeners
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()  # start to listen on a separate thread

    print(f'=== Starting experiment "{EXPERIMENT_NAME}" ({TASK_SCHEME} scheme) ===')
    if MANUAL_CTRL:
        print("Controls:")
        print("  r           reset")
        print("  space       pause / resume")
        print("  s           step (while paused)")
        print("  q           quit")
        if TASK_SCHEME == GAIT:
            print("  1-7         task: walk fwd/back, hop fwd/back/in-place, tilt, walk+tilt")
        else:
            print("  1-4         task: walk / flamingo / tilt / walk+tilt")
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
        task_scheme=TASK_SCHEME,
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
    task_vec: tuple[int, int, int] = DEFAULT_TASK_VEC  # default walk
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
            # a task key may dial in default commands (gait menu); +/- keys still adjust.
            if _cmd_set is not None:
                cmd_vel_target, cmd_tilt_target = _cmd_set
                _cmd_set = None
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


def _handle_task_key(k: str) -> bool:
    """Map a task-menu key to (bits, default cmd) for the active scheme. Returns
    True if the key was a task key. Gait keys also dial in a sensible default
    command (vel/tilt) so the gait distinction is visible; +/- keys still adjust."""
    global _task_set, _cmd_set
    if TASK_SCHEME == GAIT:
        gait_menu = {
            "1": ((1, 0, 0), (DEFAULT_VEL, 0.0), "walk forward"),
            "2": ((1, 0, 0), (-DEFAULT_VEL, 0.0), "walk backward"),
            "3": ((0, 1, 0), (DEFAULT_VEL, 0.0), "hop forward"),
            "4": ((0, 1, 0), (-DEFAULT_VEL, 0.0), "hop backward"),
            "5": ((0, 1, 0), (0.0, 0.0), "hop in place"),
            "6": ((1, 0, 0), (0.0, DEFAULT_TILT), "tilt"),
            "7": ((1, 0, 0), (DEFAULT_VEL, DEFAULT_TILT), "walk + tilt"),
        }
        if k in gait_menu:
            bits, cmd, label = gait_menu[k]
            _task_set = bits
            _cmd_set = cmd
            print(f"Task: {label}")
            return True
        return False

    # onehot menu (legacy)
    onehot_menu = {
        "1": ((1, 0, 0), "walk"),
        "2": ((0, 1, 0), "flamingo"),
        "3": ((0, 0, 1), "tilt"),
        "4": ((1, 0, 1), "walk + tilt"),
        # "5": ((0, 1, 1), "flamingo + tilt"),
    }
    if k in onehot_menu:
        bits, label = onehot_menu[k]
        _task_set = bits
        print(f"Task: {label}")
        return True
    return False


def on_press(key: Key | KeyCode | None) -> None:
    global _sim_paused, _sim_step, _sim_res
    global _left_held, _right_held, _up_held, _down_held, _zero_cmds, _task_set, _cmd_set

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
    elif k == "q":
        print("Exiting...")
        os._exit(0)
    elif k is not None:
        _handle_task_key(k)


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
