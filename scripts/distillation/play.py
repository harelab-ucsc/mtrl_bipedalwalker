import os
import warnings
from gymnasium import make
import pygame

from pynput import keyboard
from pynput.keyboard import Key, KeyCode

import torch
from utils.paths import rudin_distill_ckpt
from wrappers.bipedal_walker.distill_env import DistillEnv
from mdp.bipedal_walker.student import StudentModel

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================================
# model spec — point at a rudin[_adv]/distill/{version} run

ADVERSARIAL = True        # True -> rudin_adv/distill/...  False -> rudin/distill/...
VERSION     = "1.0.0"
MIX_MODE    = "nomix"     # "mix" or "nomix" — affects irrelevant-command behaviour
CHECKPOINT  = "best.pt"

MIX_IRRELEVANT_INPUT = MIX_MODE == "mix"
MODEL_PATH = rudin_distill_ckpt(ADVERSARIAL, VERSION, CHECKPOINT)

# =========================================

_sim_paused = False
_sim_step = False
_sim_res = False
_sim_task_delta = 0
_starting_seed = 420


def main():
    global _sim_paused, _sim_step, _sim_res, _sim_task_delta, _starting_seed

    # start key listeners
    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # start to listen on a separate thread

    print(f'=== Playing "{MODEL_PATH}" ===')
    print("Controls: w/e=prev/next task, r=reset, space=pause, s=step, q=quit")

    # load env
    print("Loading environments...")
    env = make("BipedalWalker-v3", render_mode="rgb_array")

    env = DistillEnv(
        env,
        ep_time=15,
        task_names={
            0: "walk forward",
            1: "walk backward",
            2: "flamingo",
            3: "body tilt",
            4: "walk forward + flamingo",
            5: "walk backward + flamingo",
        },
    )

    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    clock = pygame.time.Clock()

    def configureEnv(e: DistillEnv, task_id: int):
        # config the env to a certain task
        env.set_task(task_id)  # for rendering current task

        if task_id == 0:  # walk forward
            e.config_hull_reset(x_range=(0.0, 40.0))
            e.config_cmd_vel(sample_range=(0.0, 5.0), interp_time=0.5, switch_time=3, zero_prob=0.2)
            if MIX_IRRELEVANT_INPUT:
                e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
            else:
                e.config_cmd_tilt(zero_prob=1)
            active_task_bits = (1, 0, 0)
        elif task_id == 1:  # walk backward
            e.config_hull_reset(x_range=(40.0, 80.0))
            e.config_cmd_vel(sample_range=(-5.0, 0.0), interp_time=0.5, switch_time=3, zero_prob=0.2)
            if MIX_IRRELEVANT_INPUT:
                e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
            else:
                e.config_cmd_tilt(zero_prob=1)
            active_task_bits = (1, 0, 0)
        elif task_id == 2:  # flamingo
            e.config_hull_reset(x_range=(20.0, 60.0))
            if MIX_IRRELEVANT_INPUT:
                e.config_cmd_vel(sample_range=(-5.0, 5.0), switch_time=3, interp_time=0.5, zero_prob=0.2)
                e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
            else:
                e.config_cmd_vel(zero_prob=1)
                e.config_cmd_tilt(zero_prob=1)
            active_task_bits = (0, 1, 0)
        elif task_id == 3:  # tilt
            e.config_hull_reset(x_range=(20.0, 60.0))
            e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
            if MIX_IRRELEVANT_INPUT:
                e.config_cmd_vel(sample_range=(-5.0, 5.0), switch_time=3, interp_time=0.5, zero_prob=0.2)
            else:
                e.config_cmd_vel(zero_prob=1)
            active_task_bits = (0, 0, 1)
        elif task_id == 4:  # walk forward + flamingo
            e.config_hull_reset(x_range=(0.0, 40.0))
            e.config_cmd_vel(sample_range=(0.0, 5.0), interp_time=0.5, switch_time=3, zero_prob=0.2)
            if MIX_IRRELEVANT_INPUT:
                e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
            else:
                e.config_cmd_tilt(zero_prob=1)
            active_task_bits = (1, 1, 0)
        elif task_id == 5:  # walk backward + flamingo
            e.config_hull_reset(x_range=(40.0, 80.0))
            e.config_cmd_vel(sample_range=(-5.0, 0.0), interp_time=0.5, switch_time=3, zero_prob=0.2)
            if MIX_IRRELEVANT_INPUT:
                e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
            else:
                e.config_cmd_tilt(zero_prob=1)
            active_task_bits = (1, 1, 0)

        e.set_active_tasks(list(active_task_bits))
        return active_task_bits

    obs, info = env.reset(seed=_starting_seed)
    cmd_vel = info["cmd"]["x_vel"]
    cmd_tilt = info["cmd"]["tilt"]
    task_id = 0
    active_task_bits = configureEnv(env, task_id)

    # load model
    print(f'Loading model "{MODEL_PATH}"...')
    model = StudentModel()

    model.to("cpu")
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=False)["policy"])
    model.eval()

    # manually render
    def render():
        frame = env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))  # type: ignore
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        clock.tick(50)

    with torch.no_grad():
        while 1:
            pygame.event.pump()  # keep window alive on pause

            if _sim_task_delta != 0:
                task_id = (task_id + _sim_task_delta) % 6
                _sim_task_delta = 0
                _sim_res = True

            if _sim_res:
                active_task_bits = configureEnv(env, task_id)
                obs, info = env.reset(seed=_starting_seed)
                if _starting_seed is not None:
                    _starting_seed += 1
                cmd_vel = info["cmd"]["x_vel"]
                cmd_tilt = info["cmd"]["tilt"]

                _sim_res = False
                render()

                continue

            if _sim_paused:
                if not _sim_step:
                    continue
                else:
                    _sim_step = False

            assert env.action_space.shape is not None

            obs_s = StudentModel.obs(obs, task_id, cmd_vel, cmd_tilt, active_task_bits)
            action = model(torch.tensor(obs_s, dtype=torch.float32))
            obs, _, term, trunc, info = env.step(action)
            cmd_vel = info["cmd"]["x_vel"]
            cmd_tilt = info["cmd"]["tilt"]

            render()

            if term or trunc:
                _sim_res = True


def on_press(key: Key | KeyCode | None) -> None:
    global _sim_paused, _sim_step, _sim_res, _sim_task_delta

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
    elif k == "w":
        _sim_task_delta = -1
    elif k == "e":
        _sim_task_delta = 1
    elif k == "q":
        print("Exiting...")
        os._exit(0)


if __name__ == "__main__":
    main()
