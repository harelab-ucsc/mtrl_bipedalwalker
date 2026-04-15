import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
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
from wrappers.bipedal_walker.standing_env import StandReward
from wrappers.bipedal_walker.hopping_env import HopReward
from wrappers.bipedal_walker.walking_env import WalkReward
from wrappers.bipedal_walker.hopping_env_proprio import ProprioHopReward
from wrappers.bipedal_walker.walking_env_proprio import ProprioWalkReward
from wrappers.bipedal_walker.walking_backwards_proprio import ProprioWalkBackReward

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# =========================================

# PRIOR_EXP_NAME = "walk_backward/walk_backward_5_7-21_50_59-2026_04_13"
PRIOR_EXP_NAME = "walk_backward/hopped_walk_backward_7_1"
PRIOR_MODEL = "best/best_model"

EXPERIMENT_NAME = "walk_backward/walk_backward_7_6" + datetime.today().strftime(
    "-%H_%M_%S-%Y_%m_%d"
)
TIMESTEPS = 400 * 1024 * 14

# =========================================


def main():
    print("Loading environments...")

    def make_env():
        env = gym.make("BipedalWalker-v3")
        env = Monitor(
            ProprioWalkBackReward(
                env,
                ep_time=10,
                vel_sample_range=(-5, 0),
                vel_sample_zero=0.1,
                vel_switching_freq=5,
                vel_interp_speed=0.3,
            )
        )
        return env

    train_env = SubprocVecEnv([make_env for _ in range(14)])
    eval_env = SubprocVecEnv([make_env for _ in range(5)])

    # load in model from checkpoint
    prior_model_path = MODELS_DIR / f"{PRIOR_EXP_NAME}/{PRIOR_MODEL}.zip"
    model = PPO.load(
        prior_model_path,
        env=train_env,
        custom_objects={
            "learning_rate": LinearSchedule(1e-4, 3e-5, 0.8),
            "n_epochs": 25,
            "n_steps": 1024,
            "batch_size": 64,
            "ent_coef": 0.002,
        },
    )
    model.set_env(train_env)
    
    # configure logger
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
    print_run_info(train_env, model, EXPERIMENT_NAME, PRIOR_EXP_NAME, PRIOR_MODEL)

    start_time = time.time()
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=True,  # set for tensorboard
        callback=CallbackList(
            [StandardTBCallback(), RewardTermLogger(), eval_cb, ckpt_cb]
        ),
        progress_bar=True,
    )

    duration = fmt_duration(time.time() - start_time)
    print(f"Done! Total time: {duration}")
    print(f"Experiment name: {EXPERIMENT_NAME}")

    subprocess.run(
        [
            "osascript",
            "-e",
            f'display notification "Finished in {duration}" with title "Training complete" subtitle "{EXPERIMENT_NAME}"',
        ],
        check=False,
    )
    play_sound(ROOT / "assets" / "train_finish.mp3")


def print_run_info(env, model, experiment_name, prior_exp_name, prior_model_name):
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
    print(f"  continuing off from  {prior_exp_name}")
    print(f"  using model  {prior_model_name}")
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
            f"actor   in → {actor} → {act_out}",
            f"critic  in → {critic} → 1",
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
