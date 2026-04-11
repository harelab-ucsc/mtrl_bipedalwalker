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
from wrappers.bipedal_walker.sitting_env import SitReward
from wrappers.bipedal_walker.hopping_env import HopReward
from wrappers.bipedal_walker.walking_env import WalkReward
from wrappers.bipedal_walker.hopping_env_proprio import ProprioHopReward
from wrappers.bipedal_walker.walking_env_proprio import ProprioWalkReward

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# =========================================

EXPERIMENT_NAME = "sit/sit_1" + datetime.today().strftime(
    "-%H_%M_%S-%Y_%m_%d"
)
TIMESTEPS = 3_000_000

# =========================================


def main():
    print("Loading environments...")

    def make_env():
        env = gym.make("BipedalWalker-v3")
        env = Monitor(
            SitReward(
                env,
                ep_time=10,
            )
        )
        return env

    train_env = SubprocVecEnv([make_env for _ in range(14)])
    eval_env = SubprocVecEnv([make_env for _ in range(5)])

    policy_kwargs = dict(
        activation_fn=torch.nn.ELU, net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=LinearSchedule(5e-4, 3e-5, 0.5),
        n_epochs=15,
        n_steps=1024,
        batch_size=64,
        ent_coef=0.001,
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
