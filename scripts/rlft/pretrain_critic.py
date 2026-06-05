"""
scripts/rlft/pretrain_critic.py
===============================

Critic pretraining for the Rudin baseline (pure RL, no behavior cloning).

Loads a distilled student as the actor, drops a fresh critic next to it, freezes
the actor, and runs stock PPO so only the value network learns the return
landscape under the (fixed) student policy. The actor architecture is the
student's (mdp.bipedal_walker.student.HIDDEN_BC); the critic is configurable.

Which tasks the critic sees, and how often they switch, are set by the chosen
preset in pretrain_config.py (allowed_task_mixing + *_switching_time). The full
modular RLFT reward is polled by RlFTEnv itself (use_rew_for_individual_tasks).

Output is a single SB3 PPO zip (actor + critic) at
``models/rudin[_adv]/pretrained_critic/<version>/final.zip``, which
scripts/rlft/finetune.py warm-starts from.

Run:  python scripts/rlft/pretrain_critic.py --preset switching
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
from typing import OrderedDict

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure

import time
import subprocess

from utils.paths import MODELS_DIR, LOGS_DIR
from utils.logging import StandardTBCallback, RewardTermLogger, fmt_duration
from wrappers.ppo_bc.ppo_bc_env import RlFTEnv
from mdp.bipedal_walker.rlft_policy import RlFTPolicy

from pretrain_config import PRESETS, PretrainConfig

# =========================================


def make_env(cfg: PretrainConfig):
    # RlFTEnv subclasses ProprioObsWrapper, so the raw env needs no extra wrap.
    env = gym.make("BipedalWalker-v3")
    env = Monitor(
        RlFTEnv(
            env,
            ep_time=cfg.ep_time,
            cmd_switching_time=cfg.cmd_switching_time,
            task_switching_time=cfg.task_switching_time,
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


def load_student_actor(model: PPO, student_path) -> None:
    """Copy a distilled StudentModel's weights into the PPO actor (policy_net
    trunk + action head), leaving the critic at its fresh random init.

    The student is a plain Sequential of Linear/ELU then a final Linear head; its
    state_dict keys are ``policy.<idx>.{weight,bias}`` with Linears at even
    indices. RlFTPolicy's policy_net mirrors that trunk, and action_net is the
    head — so we copy trunk layers by index and the head separately. The actor
    hidden_dims MUST equal the student's (HIDDEN_BC) or this asserts.
    """
    student_sd: OrderedDict = torch.load(
        student_path, map_location="cpu", weights_only=False
    )["policy"]
    # every-other key = weight keys; their Sequential indices, trunk then head.
    layer_idx = [int(k.split(".")[1]) for k in list(student_sd.keys())[::2]]

    mlp_ext = model.policy.mlp_extractor
    action_net = model.policy.action_net

    with torch.no_grad():
        for idx in layer_idx[:-1]:
            w = student_sd[f"policy.{idx}.weight"]
            b = student_sd[f"policy.{idx}.bias"]
            assert mlp_ext.policy_net[idx].weight.shape == w.shape, (  # type: ignore
                f"actor trunk shape mismatch at layer {idx}: "
                f"{tuple(mlp_ext.policy_net[idx].weight.shape)} vs {tuple(w.shape)} "  # type: ignore
                f"— hidden_dims must equal the student's HIDDEN_BC."
            )
            mlp_ext.policy_net[idx].weight.copy_(w)  # type: ignore
            mlp_ext.policy_net[idx].bias.copy_(b)  # type: ignore
        head_w = student_sd[f"policy.{layer_idx[-1]}.weight"]
        head_b = student_sd[f"policy.{layer_idx[-1]}.bias"]
        assert action_net.weight.shape == head_w.shape, (
            f"action head shape mismatch: {tuple(action_net.weight.shape)} vs "
            f"{tuple(head_w.shape)}"
        )
        action_net.weight.copy_(head_w)
        action_net.bias.copy_(head_b)


def main(cfg: PretrainConfig):
    assert cfg.load_student_from, "cfg.load_student_from is not set — point it at a distilled student .pt."
    student_path = MODELS_DIR / cfg.load_student_from
    assert student_path.exists(), f"distilled student not found: {student_path}"

    experiment_name = cfg.experiment_name
    t0 = time.time()

    print("Loading environments...")
    env_fn = partial(make_env, cfg)
    train_env = SubprocVecEnv([env_fn for _ in range(cfg.n_train_envs)])

    print("Creating model...")
    model = PPO(
        RlFTPolicy,
        env=train_env,
        policy_kwargs=dict(
            hidden_dims=list(cfg.hidden_dims),
            critic_hidden_dims=list(cfg.critic_hidden_dims),
            activation_fn=cfg.activation_fn,
        ),
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

    print(f"Loading student actor from {student_path}...")
    load_student_actor(model, student_path)

    # Freeze actor trunk + head + log_std so only the value network learns.
    for p in model.policy.mlp_extractor.policy_net.parameters():
        p.requires_grad_(False)
    for p in model.policy.action_net.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        model.policy.log_std.fill_(cfg.init_log_std)
    model.policy.log_std.requires_grad_(False)

    model.set_logger(configure(str(LOGS_DIR / experiment_name), ["tensorboard"]))
    train_env.reset()

    ckpt_cb = CheckpointCallback(
        save_freq=max(500000 // train_env.num_envs, 1),
        save_path=f"{MODELS_DIR}/{experiment_name}/",
    )

    print_run_info(cfg, train_env, model, experiment_name, student_path)

    print(f"Starting critic pretraining ({cfg.timesteps:,} timesteps)...")
    start = time.time()
    model.learn(
        total_timesteps=cfg.timesteps,
        reset_num_timesteps=False,
        callback=CallbackList([StandardTBCallback(), RewardTermLogger(), ckpt_cb]),
        progress_bar=True,
    )

    final_path = f"{MODELS_DIR}/{experiment_name}/final.zip"
    model.save(final_path)
    print(f"Saved pretrained critic (actor + critic) to {final_path}")

    duration = fmt_duration(time.time() - start)
    print(f"Done! Total time: {duration}  (since launch: {fmt_duration(time.time() - t0)})")
    print(f"Experiment name: {experiment_name}")

    try:
        subprocess.run(
            [
                "osascript", "-e",
                f'display notification "Finished in {duration}" with title "Critic pretrain complete" subtitle "{experiment_name}"',
            ],
            check=False,
        )
    except FileNotFoundError:
        pass  # not on macOS

    train_env.close()


def print_run_info(cfg, env, model, experiment_name, student_path):
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

    def switch_desc(t: float) -> str:
        return f"{t}s" if t < cfg.ep_time else f"{t}s (off, >= ep_time)"

    print(f"\n{'=' * 44}")
    print(f"  critic_pretrain  {experiment_name}")
    print(f"  frozen actor     {student_path}")
    print("  Note: actor is frozen; only the value network trains.")
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
        "ppo",
        [
            f"device            {model.device}",
            f"lr                {cfg.learning_rate}",
            f"n_steps           {model.n_steps}",
            f"batch_size        {model.batch_size}",
            f"n_epochs          {model.n_epochs}",
            f"gamma / lambda    {model.gamma} / {model.gae_lambda}",
            f"vf_coef / ent     {model.vf_coef} / {model.ent_coef}",
            f"clip_range        {cfg.clip_range}",
            f"init_log_std      {cfg.init_log_std:.4f}",
        ],
    )

    def net_summary(net):
        return " -> ".join(str(l.out_features) for l in net if hasattr(l, "out_features"))

    actor = net_summary(p.mlp_extractor.policy_net)
    critic = net_summary(p.mlp_extractor.value_net)
    section(
        "network",
        [
            f"actor  (frozen)  in -> {actor} -> {p.action_net.out_features}",
            f"critic (train)   in -> {critic} -> 1",
            f"activation       {cfg.activation_fn.__name__}",
        ],
    )

    print(f"\n{'=' * 44}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a Rudin-baseline critic from a named preset.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="switching",
        help="which pretrain_config preset to run (default: switching)",
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
    if args.timesteps is not None:
        from dataclasses import replace
        cfg = replace(cfg, timesteps=args.timesteps)
    print(f"Using preset: {args.preset}  ->  experiment {cfg.experiment_name}")
    main(cfg)
