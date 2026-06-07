"""
scripts/ppo_scratch/train.py
============================

From-scratch PPO floor baseline for the BC_Coef ablation.

Trains a stock ``stable_baselines3.PPO`` from random init, in ONE shot, on the
full task union (5 single + 2 combination gait tasks, fast 3s switching) using the
same modular reward (``RlFTEnv`` with ``use_rew_for_individual_tasks=True``) and the
same actor/critic architecture as the ppo_bc method. There is NO behavior cloning,
NO experts, NO curriculum, NO warm-start and NO pretrained critic — this is the
naive-PPO reference everything else is measured against.

Fairness: env, reward, network (actor 512,256,256,128,64 / critic 1024,512,512,256,256,
ELU, log_std=log(1)) and most PPO knobs are identical to the comb_switching RL stage.
The differences are intrinsic to "from scratch": stock-PPO clip/ent (0.2 / 0.002), a
from-scratch LR schedule (5e-4->3e-5, since the finetune 5e-5 can't learn from random
init), and 2x the timesteps (8.6M) to offset the missing pretrain/critic stages.

Run:  python scripts/ppo_scratch/train.py --preset comb_switching
      python scripts/ppo_scratch/train.py --preset comb_switching --timesteps 28672   # smoke test
"""

# Cap per-process CPU thread pools before importing numpy/torch. Each
# SubprocVecEnv worker is a separate process, and by default each one's
# numpy/MKL/OpenMP spins up one thread per core — so N workers oversubscribe the
# box (load average >> core count) and thrash on spin-waits instead of doing
# work. Parallelism here is across processes, so 1 BLAS thread per process is
# correct. Override the cap by exporting OMP_NUM_THREADS=... before launching.
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from functools import partial

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure

import time
import subprocess

# NOTE: deliberately NO pygame / pynput imports anywhere in this file — they break
# `import` on a headless Linux box (no $DISPLAY). End-of-run notification is
# osascript-only (macOS), wrapped in try/except.

from utils.paths import MODELS_DIR, LOGS_DIR
from utils.logging import StandardTBCallback, RewardTermLogger, fmt_duration
from wrappers.ppo_bc.ppo_bc_env import RlFTEnv

from train_config import PRESETS, TrainConfig

# =========================================


def make_env(cfg: TrainConfig):
    # RlFTEnv is a plain gymnasium wrapper (policy-agnostic); the modular reward is
    # computed inside it. Identical construction to scripts/rlft/finetune.py.
    env = gym.make("BipedalWalker-v3")
    env = Monitor(
        RlFTEnv(
            env,
            ep_time=cfg.ep_time,
            cmd_switching_time=cfg.cmd_switching_time,
            task_switching_time=cfg.task_switching_time,
            task_switch_replacement=cfg.task_switch_replacement,
            cmd_interp_speed=cfg.cmd_interp_speed,
            cmd_sample_range=cfg.cmd_sample_range,
            cmd_sample_zero=cfg.cmd_sample_zero,
            allowed_task_mixing=cfg.allowed_task_mixing,
            use_rew_for_individual_tasks=cfg.use_indv_task_rew,
            hull_x_range=cfg.hull_x_range,
            task_scheme=cfg.task_scheme,
        )
    )
    return env


def main(cfg: TrainConfig, timesteps: int | None = None):
    total_timesteps = timesteps if timesteps is not None else cfg.timesteps
    experiment_name = cfg.experiment_name

    print("Loading environments...")
    env_fn = partial(make_env, cfg)
    train_env = SubprocVecEnv([env_fn for _ in range(cfg.n_train_envs)])
    eval_env = SubprocVecEnv([env_fn for _ in range(cfg.n_eval_envs)])

    # Stock PPO from random init. net_arch=dict(pi=..., vf=...) builds the same
    # independent actor/critic trunks as PpoBcNetwork; log_std_init=log(1) matches
    # the state-independent std the method pins.
    policy_kwargs = dict(
        net_arch=dict(pi=list(cfg.hidden_dims), vf=list(cfg.critic_hidden_dims)),
        activation_fn=cfg.activation_fn,
        log_std_init=cfg.log_std_init,
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=cfg.learning_rate,
        n_epochs=cfg.n_epochs,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        max_grad_norm=cfg.max_grad_norm,
        device=cfg.device,
        verbose=0,
    )

    model.set_logger(configure(str(LOGS_DIR / experiment_name), ["tensorboard"]))
    train_env.reset()

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODELS_DIR}/{experiment_name}/best",
        eval_freq=max(50000 // train_env.num_envs, 1),
        n_eval_episodes=5,
        verbose=0,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(100000 // train_env.num_envs, 1),
        save_path=f"{MODELS_DIR}/{experiment_name}/",
    )
    callbacks = [StandardTBCallback(), RewardTermLogger(), eval_cb, ckpt_cb]

    print_run_info(cfg, train_env, model, experiment_name, total_timesteps)

    print(f"Starting from-scratch PPO ({total_timesteps:,} timesteps)...")
    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=True,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    model.save(f"{MODELS_DIR}/{experiment_name}/final")

    train_env.close()
    eval_env.close()

    duration = fmt_duration(time.time() - start)
    print(f"Done! Total time: {duration}")
    print(f"Experiment name: {experiment_name}")

    try:
        subprocess.run(
            [
                "osascript", "-e",
                f'display notification "Finished in {duration}" with title "PPO baseline complete" subtitle "{experiment_name}"',
            ],
            check=False,
        )
    except FileNotFoundError:
        pass  # not on macOS


def print_run_info(cfg: TrainConfig, env, model, experiment_name, total_timesteps):
    env_id = env.get_attr("spec")[0].id
    obs = env.observation_space
    act = env.action_space
    p = model.policy

    def section(title, lines):
        print(f"\n  {title}")
        print(f"  {'-' * 40}")
        for line in lines:
            print(f"    {line}")

    def task_name(t) -> str:
        name = getattr(t, "name", None)
        if name is not None:
            return name
        bits = tuple(int(x) for x in t)
        return "+".join(q for b, q in zip(bits, ("walk", "flamingo", "tilt")) if b) or "idle"

    def lr_desc(lr) -> str:
        if callable(lr):
            return f"sched {lr(1.0):.1e} -> {lr(0.0):.1e}"
        return f"{lr:.1e}"

    def switch_desc(t: float) -> str:
        return f"{t}s" if t < cfg.ep_time else f"{t}s (off, >= ep_time)"

    print(f"\n{'=' * 44}")
    print(f"  ppo_scratch (from random init)  {experiment_name}")
    print(f"  timesteps        {total_timesteps:,}")
    print(f"  Note: no BC / experts / curriculum / warm-start / pretrained critic.")
    print(f"{'=' * 44}")

    cmd_vel_sw, cmd_tilt_sw = cfg.cmd_switching_time
    section(
        "environment",
        [
            f"{env.num_envs}x  {env_id}",
            f"obs  {obs.shape}  {obs.dtype}",
            f"act  {act.shape}  [{act.low[0]:.1f}, {act.high[0]:.1f}]",
            f"ep_time             {cfg.ep_time}s",
            f"cmd_switch vel/tilt {switch_desc(cmd_vel_sw)} / {switch_desc(cmd_tilt_sw)}",
            f"task_switch         {switch_desc(cfg.task_switching_time)}",
            f"tasks               {', '.join(task_name(t) for t in cfg.allowed_task_mixing)}",
            f"use_indv_task_rew   {cfg.use_indv_task_rew}",
        ],
    )

    section(
        "ppo (from scratch)",
        [
            f"device            {model.device}",
            f"lr                {lr_desc(cfg.learning_rate)}",
            f"clip_range        {cfg.clip_range}",
            f"n_steps           {model.n_steps}",
            f"batch_size        {model.batch_size}",
            f"n_epochs          {model.n_epochs}",
            f"gamma / lambda    {model.gamma} / {model.gae_lambda}",
            f"vf_coef / ent     {model.vf_coef} / {model.ent_coef}",
            f"log_std_init      {cfg.log_std_init}",
        ],
    )

    def net_summary(net):
        return " -> ".join(str(l.out_features) for l in net if hasattr(l, "out_features"))

    actor = net_summary(p.mlp_extractor.policy_net)
    critic = net_summary(p.mlp_extractor.value_net)
    std = f"{torch.exp(p.log_std).mean().item():.3f}" if hasattr(p, "log_std") else "n/a"
    section(
        "network",
        [
            f"actor   in -> {actor} -> {p.action_net.out_features}",
            f"critic  in -> {critic} -> 1",
            f"activation          {cfg.activation_fn.__name__}",
            f"action std (init)   {std}",
        ],
    )

    print(f"\n{'=' * 44}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a from-scratch PPO baseline from a named preset.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="comb_switching",
        help="which train_config preset to run (default: comb_switching)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="override cfg.timesteps (useful for quick smoke tests)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PRESETS[args.preset]
    print(f"Using preset: {args.preset}  ->  experiment {cfg.experiment_name}")
    main(cfg, args.timesteps)
