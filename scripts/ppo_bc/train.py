import gymnasium as gym
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure

from datetime import datetime
import time
import os
import subprocess
import threading

import pygame
from pynput import keyboard as kb

from utils.paths import MODELS_DIR, LOGS_DIR, ROOT
from utils.logging import StandardTBCallback, RewardTermLogger, fmt_duration

# new env that drives task + cmd sampling internally (walk / flamingo / tilt mix).
from wrappers.ppo_bc.ppo_bc_env import RlFTEnv

# local copy of the sb3 surface we will extend with bc / dagger later.
from ppo_bc_sb3 import PPO_BC, PpoBcPolicy

if not os.path.exists(MODELS_DIR / "ppo_bc"):
    os.makedirs(MODELS_DIR / "ppo_bc")

if not os.path.exists(LOGS_DIR / "ppo_bc"):
    os.makedirs(LOGS_DIR / "ppo_bc")

# =========================================

EXPERIMENT_NAME = "ppo_bc/1.0.0" + datetime.today().strftime(
    "-%H_%M_%S-%Y_%m_%d"
)
TIMESTEPS = 200 * 1024 * 14

# --- environment params (mirrors scripts/rlft/finetune_v2.py defaults) ---
N_TRAIN_ENVS         = 14
N_EVAL_ENVS          = 5
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
HULL_X_RANGE         = (40.0, 80.0)

# =========================================


def make_env():
    # RlFTEnv subclasses ProprioObsWrapper internally, so we don't need to wrap
    # the raw bipedal walker env in ProprioObsWrapper ourselves.
    env = gym.make("BipedalWalker-v3")
    env = Monitor(
        RlFTEnv(
            env,
            ep_time=EP_TIME,
            cmd_switching_time=CMD_SWITCHING_TIME,
            task_switching_time=TASK_SWITCHING_TIME,
            cmd_interp_speed=CMD_INTERP_SPEED,
            cmd_sample_range=CMD_SAMPLE_RANGE,
            cmd_sample_zero=CMD_SAMPLE_ZERO,
            allowed_task_mixing=ALLOWED_TASK_MIXING,
            hull_x_range=HULL_X_RANGE,
        )
    )
    return env


def main():
    print("Loading environments...")

    train_env = SubprocVecEnv([make_env for _ in range(N_TRAIN_ENVS)])
    eval_env = SubprocVecEnv([make_env for _ in range(N_EVAL_ENVS)])

    # policy_kwargs map onto PpoBcPolicy.__init__. hidden_dims is the actor
    # trunk; critic_hidden_dims is the critic trunk (defaults to hidden_dims if
    # not specified). activation_fn is forwarded by ActorCriticPolicy.
    policy_kwargs = dict(
        hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation_fn=torch.nn.ELU,
    )

    # PPO_BC currently behaves identically to sb3 PPO. all loss / rollout hooks
    # live in src/ppo_bc_sb3/ so dagger and the bc term can be added by editing
    # those files without touching this script.
    model = PPO_BC(
        PpoBcPolicy,
        train_env,
        verbose=0,
        learning_rate=LinearSchedule(5e-4, 3e-5, 0.8),
        n_epochs=15,
        n_steps=1024,
        batch_size=64,
        ent_coef=0.002,
        policy_kwargs=policy_kwargs,
        device=torch.device("cpu"),
    )
    model.set_logger(configure(str(LOGS_DIR / EXPERIMENT_NAME), ["tensorboard"]))
    train_env.reset()

    # define callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{str(MODELS_DIR)}/{EXPERIMENT_NAME}/best",
        eval_freq=max(50000 // train_env.num_envs, 1),
        n_eval_episodes=5,
        verbose=0,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(100000 // train_env.num_envs, 1),
        save_path=f"{MODELS_DIR}/{EXPERIMENT_NAME}/",
    )

    # print out model and environment settings
    print_run_info(train_env, model, EXPERIMENT_NAME)

    start_time = time.time()
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        callback=CallbackList(
            [StandardTBCallback(), RewardTermLogger(), eval_cb, ckpt_cb]
        ),
        progress_bar=True,
    )


    duration = fmt_duration(time.time() - start_time)
    print(f"Done! Total time: {duration}")
    print(f"Experiment name: {EXPERIMENT_NAME}")

    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "Finished in {duration}" with title "Training complete" subtitle "{EXPERIMENT_NAME}"'],
            check=False,
        )
    except FileNotFoundError:
        pass  # not on macOS
    try:
        play_sound(ROOT / "assets" / "train_finish.mp3")
    except Exception as e:
        print(f"(skipping play_sound: {e})")


def print_run_info(env, model, experiment_name):
    env_id = env.get_attr("spec")[0].id
    obs = env.observation_space
    act = env.action_space
    p = model.policy

    def section(title, lines):
        print(f"\n  {title}")
        print(f"  {'-' * 40}")
        for line in lines:
            print(f"    {line}")

    print(f"\n{'=' * 44}")
    print(f"  experiment  {experiment_name}")
    print(f"{'=' * 44}")

    section(
        "environment",
        [
            f"{env.num_envs}x  {env_id}",
            f"obs  {obs.shape}  {obs.dtype}",
            f"act  {act.shape}  [{act.low[0]:.1f}, {act.high[0]:.1f}]",
        ],
    )

    section(
        "policy",
        [
            f"device            {model.device}",
            f"n_steps           {model.n_steps}",
            f"batch_size        {model.batch_size}",
            f"n_epochs          {model.n_epochs}",
            f"gamma             {model.gamma}",
            f"lambda            {model.gae_lambda}",
            f"target_kl         {model.target_kl}",
            f"entropy_coef      {model.ent_coef}",
            f"stats_win_size    {model._stats_window_size}",
            f"lr                {model.learning_rate}",
            f"seed              {model.seed}",
        ],
    )

    # extract layer sizes from mlp_extractor
    def net_summary(net):
        sizes = [str(l.out_features) for l in net if hasattr(l, "out_features")]
        return " -> ".join(sizes)

    actor = net_summary(p.mlp_extractor.policy_net)
    critic = net_summary(p.mlp_extractor.value_net)
    act_out = p.action_net.out_features
    section(
        "network",
        [
            f"actor   in -> {actor} -> {act_out}",
            f"critic  in -> {critic} -> 1",
        ],
    )

    print(f"\n{'=' * 44}\n")


def play_sound(path):
    pygame.mixer.init()
    pygame.mixer.music.load(str(path))
    pygame.mixer.music.play()
    print("Tip: Press Esc to stop the sound.")

    stop = threading.Event()

    def on_press(key):
        if key == kb.Key.esc:
            stop.set()

    listener = kb.Listener(on_press=on_press)
    listener.start()
    while pygame.mixer.music.get_busy() and not stop.is_set():
        time.sleep(0.1)
    listener.stop()

    pygame.mixer.music.stop()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
