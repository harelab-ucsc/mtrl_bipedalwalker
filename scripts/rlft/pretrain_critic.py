from typing import Callable, OrderedDict, cast

import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
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

from torch import nn
from utils.paths import MODELS_DIR, LOGS_DIR, ROOT
from utils.logging import StandardTBCallback, RewardTermLogger, fmt_duration
from wrappers.bipedal_walker.rltf_env import RlFTEnv

Schedule = Callable[[float], float]

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# =========================================

DISTILLED_STUDENT = "distill/ml/best.pt"
EXPERIMENT_NAME = "rlft/pretrain/ml"
TIMESTEPS = 100 * 1024 * 14  # pretraining needs a lot less timesteps

# =========================================


def main():
    wall_clk_start = time.time()
    
    print(f"[{(time.time() - wall_clk_start):.2f}s] Loading environments...")

    def make_env():
        env = gym.make("BipedalWalker-v3")
        env = Monitor(
            RlFTEnv(
                env,
                ep_time=10,
                vel_sample_range=(0, 5),
                vel_sample_zero=0.2,
                vel_interp_speed=0.3,
            )
        )
        return env

    train_env = SubprocVecEnv([make_env for _ in range(14)])
    eval_env = SubprocVecEnv([make_env for _ in range(5)])

    # create custom model, and preload the policy net
    class RlFTNetwork(nn.Module):
        def __init__(
            self,
            feature_dim: int,
            hidden_dims: tuple,
            activation: type,
        ):
            super().__init__()

            def _make_net(in_dim, hidden):
                layers, d = [], in_dim
                for h in hidden:
                    layers += [nn.Linear(d, h), activation()]
                    d = h
                return nn.Sequential(*layers), d

            self.policy_net, self.latent_dim_pi = _make_net(feature_dim, hidden_dims)
            self.value_net, self.latent_dim_vf = _make_net(feature_dim, [512, 256, 128])

        def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
                If all layers are shared, then ``latent_policy == latent_value``
            """
            return self.forward_actor(features), self.forward_critic(features)

        def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
            return self.policy_net(features)

        def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
            return self.value_net(features)

    class RlFTPolicy(ActorCriticPolicy):
        def __init__(
            self,
            obs_space: spaces.Space,
            act_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            hidden_dims,
            **kwargs,
        ):
            self._hidden_dims = hidden_dims
            kwargs["ortho_init"] = False
            super().__init__(obs_space, act_space, lr_schedule, **kwargs)

        def _build_mlp_extractor(self) -> None:
            self.mlp_extractor = RlFTNetwork(
                self.features_dim,
                self._hidden_dims,
                self.activation_fn,
            )

    # initialize model
    print(f"[{(time.time() - wall_clk_start):.2f}s] Creating model...")

    model = PPO(
        RlFTPolicy,
        env=train_env,
        policy_kwargs=dict(hidden_dims=[320, 160, 80], activation_fn=torch.nn.ELU),
        learning_rate=1e-3,
        n_epochs=10,
        n_steps=1024,
        batch_size=512,
        ent_coef=0,
        vf_coef=1.0  # fine cuz we froze actor
    )
    # model.set_env(train_env)

    # load in actor weights and freeze them
    print(f"[{(time.time() - wall_clk_start):.2f}s] Preloading policy...")

    student_path = MODELS_DIR / DISTILLED_STUDENT
    student_sd: OrderedDict = torch.load(
        student_path, map_location="cpu", weights_only=False
    )["policy"]
    layer_idx = [int(i.split(".")[1]) for i in list(student_sd.keys())[::2]]

    sb3_policy = model.policy
    mlp_ext = sb3_policy.mlp_extractor
    action_net = sb3_policy.action_net

    # copy weights
    with torch.no_grad():
        # mlp extractor
        for idx in layer_idx[:-1]:
            mlp_ext.policy_net[idx].weight.copy_(student_sd[f"policy.{idx}.weight"])  # type: ignore
            mlp_ext.policy_net[idx].bias.copy_(student_sd[f"policy.{idx}.bias"])  # type: ignore
        # actor net
        action_net.weight.copy_(student_sd[f"policy.{layer_idx[-1]}.weight"])  # type: ignore
        action_net.bias.copy_(student_sd[f"policy.{layer_idx[-1]}.bias"])  # type: ignore

    # freeze weights and logstd
    for p in mlp_ext.policy_net.parameters():
        p.requires_grad_(False)
    for p in action_net.parameters():
        p.requires_grad_(False)
    with torch.no_grad():  # initialize log_std to something small
        model.policy.log_std.fill_(-3.69)  # std approx 0.025,
    sb3_policy.log_std.requires_grad_(False)

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
    print_run_info(
        train_env, model, EXPERIMENT_NAME, DISTILLED_STUDENT
    )

    start_time = time.time()
    model.learn(
        total_timesteps=TIMESTEPS,
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
            f'display notification "Finished in {duration}" with title "Pretraining complete" subtitle "{EXPERIMENT_NAME}"',
        ],
        check=False,
    )
    play_sound(ROOT / "assets" / "train_finish.mp3")


def print_run_info(env, model, experiment_name, distilled_student_name):
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
    print(f"  pretraining         {experiment_name}")
    print(f"  using frozen policy {distilled_student_name}")
    print(f"  Note: this script keeps the policy frozen and just pretrain the value function.")
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
        f"network",
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
