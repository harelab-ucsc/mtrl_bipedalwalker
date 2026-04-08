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

from utils.paths import MODELS_DIR, LOGS_DIR
from utils.logging import StandardTBCallback, RewardTermLogger, fmt_duration
# from wrappers.bipedal_walker.standing_env import StandReward
from wrappers.bipedal_walker.hopping_env import HopReward

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# =========================================

EXPERIMENT_NAME = "hop_5" + datetime.today().strftime("-%H_%M_%S-%Y_%m_%d")
TIMESTEPS = 200 * 2048 * 14

# =========================================


def main():
    print("Loading environments...")

    def make_env():
        env = gym.make("BipedalWalker-v3")
        env = Monitor(HopReward(env, vel_sample_zero=0.05))
        return env

    train_env = SubprocVecEnv([make_env for _ in range(14)])
    eval_env = SubprocVecEnv([make_env for _ in range(5)])
    
    # train_env = make_vec_env(
    #     "BipedalWalker-v3",
    #     n_envs=14,
    #     vec_env_cls=SubprocVecEnv
    # )
    # eval_env = make_vec_env(
    #     "BipedalWalker-v3",
    #     n_envs=5,
    #     vec_env_cls=SubprocVecEnv
    # )

    policy_kwargs = dict(
        activation_fn=torch.nn.ELU,
        net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=LinearSchedule(5e-4, 3e-5, 0.5),
        n_epochs=15,
        batch_size=64,
        ent_coef=0.005,
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
        callback=CallbackList([StandardTBCallback(), RewardTermLogger(), eval_cb, ckpt_cb]),
        progress_bar=True,
    )

    print(f"Done! Total time: {fmt_duration(time.time() - start_time)}")
    print(f"Experiment name: {EXPERIMENT_NAME}")


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


if __name__ == "__main__":
    main()
