"""
Critic pretraining for PPO_BC.

Loads a trained PPO_BC actor, drops a fresh critic next to it, and runs PPO
with the actor frozen so only the value network learns. Output is a PPO_BC
zip that train.py can resume via ``load_model``.

All hyperparameters live in pretrain_critic_config.py; pick one with --preset.
"""

import argparse
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure

import time
import os
import subprocess

from utils.paths import MODELS_DIR, LOGS_DIR
from utils.logging import StandardTBCallback, RewardTermLogger, fmt_duration

from wrappers.ppo_bc.ppo_bc_env import RlFTEnv
from ppo_bc_sb3 import PPO_BC, PpoBcPolicy, load_expert
from mdp.bipedal_walker.tasks import GAIT

# all hyperparameters live in pretrain_critic_config.py; pick one with --preset.
from pretrain_critic_config import PRESETS, PretrainCriticConfig

if not os.path.exists(MODELS_DIR / "ppo_bc"):
    os.makedirs(MODELS_DIR / "ppo_bc")
if not os.path.exists(LOGS_DIR / "ppo_bc"):
    os.makedirs(LOGS_DIR / "ppo_bc")

# =========================================


def build_experts(cfg: PretrainCriticConfig):
    """Load expert checkpoints once and return a dict[task_name, callable(obs)->act].

    Mirrors train.py.build_experts (scheme-aware keys). Required by the PPO_BC ctor
    but never polled here (collect_data=False), so it just needs to be present and
    consistent with the active scheme's directional task names.
    """
    n_proprio = cfg.n_proprio

    def _vel_call(expert):
        def call(obs):
            cmd_vel = obs[:, n_proprio : n_proprio + 1]
            x = np.concatenate([obs[:, :n_proprio], cmd_vel], axis=-1)  # [N, 15]
            return expert.predict(x, deterministic=True)[0]
        return call

    def _tilt_call(expert):
        def call(obs):
            cmd_tilt = obs[:, n_proprio + 1 : n_proprio + 2]
            x = np.concatenate([obs[:, :n_proprio], cmd_tilt], axis=-1)
            return expert.predict(x, deterministic=True)[0]
        return call

    # expert_paths holds bare paths; prefix with MODELS_DIR here (config stays bare).
    walk_fwd = load_expert(MODELS_DIR / cfg.expert_paths["walk_forward"])
    walk_bwd = load_expert(MODELS_DIR / cfg.expert_paths["walk_backward"])
    tilt = load_expert(MODELS_DIR / cfg.expert_paths["body_tilt"])

    if cfg.task_scheme == GAIT:
        hop_fwd = load_expert(MODELS_DIR / cfg.expert_paths["hop_forward"])
        hop_bwd = load_expert(MODELS_DIR / cfg.expert_paths["hop_backward"])
        return {
            "walk_forward": _vel_call(walk_fwd),
            "walk_backward": _vel_call(walk_bwd),
            "hop_forward": _vel_call(hop_fwd),
            "hop_backward": _vel_call(hop_bwd),
            "tilt": _tilt_call(tilt),
        }

    hop_fwd = load_expert(MODELS_DIR / cfg.expert_paths["hop_forward"])

    def flamingo_call(obs):
        zero = np.zeros((obs.shape[0], 1), dtype=obs.dtype)
        x = np.concatenate([obs[:, :n_proprio], zero], axis=-1)
        return hop_fwd.predict(x, deterministic=True)[0]

    return {
        "walk_forward": _vel_call(walk_fwd),
        "walk_backward": _vel_call(walk_bwd),
        "flamingo": flamingo_call,
        "tilt": _tilt_call(tilt),
    }


def make_env(cfg: PretrainCriticConfig):
    # RlFTEnv subclasses ProprioObsWrapper internally, so we don't need to wrap
    # the raw bipedal walker env in ProprioObsWrapper ourselves.
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


def load_actor_only(model: PPO_BC, zip_path) -> None:
    """Copy actor weights (policy_net trunk + action_net + log_std) from a
    saved PPO_BC zip into `model`, leaving the critic at its random init.

    PPO_BC.load reconstructs the source policy using *its own* saved
    policy_kwargs, so the source critic dims don't have to match the
    destination's critic_hidden_dims. The actor dims do have to match — we
    assert layer-by-layer.
    """
    src = PPO_BC.load(
        zip_path,
        experts=model.experts,
        task_bits=model.task_bits,
        task_scheme=model.task_scheme,
        device="cpu",
    )
    src_sd = src.policy.state_dict()
    dst_sd = model.policy.state_dict()
    for k in list(dst_sd.keys()):
        # skip critic trunk + value head; keep the freshly-init critic
        if k.startswith("mlp_extractor.value_net") or k.startswith("value_net"):
            continue
        if k in src_sd:
            assert dst_sd[k].shape == src_sd[k].shape, (
                f"actor weight shape mismatch on {k}: dst {tuple(dst_sd[k].shape)} "
                f"vs src {tuple(src_sd[k].shape)} — hidden_dims must match the actor zip."
            )
            dst_sd[k] = src_sd[k]
    model.policy.load_state_dict(dst_sd)


def main(cfg: PretrainCriticConfig):
    assert cfg.load_actor_from, "cfg.load_actor_from is not set — point it at a PPO_BC zip."
    actor_path = Path(MODELS_DIR / cfg.load_actor_from)
    assert actor_path.exists(), f"load_actor_from does not exist: {actor_path}"

    print("Loading environments...")

    # bind cfg into the env factory so SubprocVecEnv workers (spawned, not forked)
    # reconstruct the same config without relying on module globals.
    env_fn = partial(make_env, cfg)
    train_env = SubprocVecEnv([env_fn for _ in range(cfg.n_train_envs)])

    print("Loading experts (ctor requirement only; not polled)...")
    experts = build_experts(cfg)

    policy_kwargs = dict(
        hidden_dims=list(cfg.hidden_dims),
        critic_hidden_dims=list(cfg.critic_hidden_dims),
        activation_fn=cfg.activation_fn,
    )

    # Fresh PPO_BC. collect_data=False disables DAgger relabeling + dataset
    # growth entirely; bc_coef=0 makes the BC term explicitly inert.
    model = PPO_BC(
        PpoBcPolicy,
        train_env,
        experts=experts,
        task_bits=cfg.task_bits,
        task_scheme=cfg.task_scheme,
        act_var_floor=cfg.act_var_floor,
        bc_coef=0.0,
        collect_data=False,
        verbose=0,
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
        policy_kwargs=policy_kwargs,
        device=cfg.device,
    )

    print(f"Loading actor weights from {actor_path}...")
    load_actor_only(model, actor_path)

    # Freeze actor + action head + log_std so only the critic learns.
    for p in model.policy.mlp_extractor.policy_net.parameters():
        p.requires_grad_(False)
    for p in model.policy.action_net.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        model.policy.log_std.fill_(cfg.init_log_std)
    model.policy.log_std.requires_grad_(False)

    model.set_logger(configure(str(LOGS_DIR / cfg.experiment_name), ["tensorboard"]))
    train_env.reset()

    ckpt_cb = CheckpointCallback(
        save_freq=max(500000 // train_env.num_envs, 1),
        save_path=f"{MODELS_DIR}/{cfg.experiment_name}/",
    )

    print_run_info(cfg, train_env, model)

    print(f"Starting critic pretraining ({cfg.timesteps:,} timesteps)...")
    start = time.time()
    model.learn(
        total_timesteps=cfg.timesteps,
        reset_num_timesteps=False,
        callback=CallbackList([StandardTBCallback(), RewardTermLogger(), ckpt_cb]),
        progress_bar=True,
    )

    final_path = f"{MODELS_DIR}/{cfg.experiment_name}/final.zip"
    model.save(final_path)
    print(f"Saved final model to {final_path}")

    duration = fmt_duration(time.time() - start)
    print(f"Done! Total time: {duration}")
    print(f"Experiment name: {cfg.experiment_name}")

    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "Finished in {duration}" with title "Critic pretrain complete" subtitle "{cfg.experiment_name}"',
            ],
            check=False,
        )
    except FileNotFoundError:
        pass  # not on macOS


def print_run_info(cfg: PretrainCriticConfig, env, model):
    """Echo the full resolved config (+ the actually-built model/env) so a run's
    settings can be eyeballed before training — note the actor is frozen."""
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
        return "+".join(p for b, p in zip(bits, ("walk", "flamingo", "tilt")) if b) or "idle"

    def lr_desc(lr) -> str:
        # schedules take progress_remaining in [0, 1]: 1.0 = start, 0.0 = end.
        if callable(lr):
            return f"sched {lr(1.0):.1e} -> {lr(0.0):.1e}"
        return f"{lr:.1e}"

    def switch_desc(t: float) -> str:
        return f"{t}s" if t < cfg.ep_time else f"{t}s (off, >= ep_time)"

    print(f"\n{'=' * 44}")
    print(f"  critic_pretrain  {cfg.experiment_name}")
    print(f"  timesteps        {cfg.timesteps:,}")
    print(f"  frozen actor     {cfg.load_actor_from}")
    print(f"  Note: actor is frozen; only the value network trains.")
    print(f"{'=' * 44}")

    cmd_vel_sw, cmd_tilt_sw = cfg.cmd_switching_time
    section(
        "environment",
        [
            f"train envs          {cfg.n_train_envs}",
            f"env                 {env_id}",
            f"obs                 {obs.shape}  {obs.dtype}",
            f"act                 {act.shape}  [{act.low[0]:.1f}, {act.high[0]:.1f}]",
            f"ep_time             {cfg.ep_time}s",
            f"cmd_switch vel/tilt {switch_desc(cmd_vel_sw)} / {switch_desc(cmd_tilt_sw)}",
            f"task_switch         {switch_desc(cfg.task_switching_time)}",
            f"cmd_range vel/tilt  {cfg.cmd_sample_range[0]} / {cfg.cmd_sample_range[1]}",
            f"cmd_zero_p vel/tilt {cfg.cmd_sample_zero[0]} / {cfg.cmd_sample_zero[1]}",
            f"cmd_interp vel/tilt {cfg.cmd_interp_speed[0]} / {cfg.cmd_interp_speed[1]}",
            f"hull_x_range        {cfg.hull_x_range}",
        ],
    )

    section(
        "task / reward",
        [
            f"tasks               {', '.join(task_name(t) for t in cfg.allowed_task_mixing)}",
            f"use_indv_task_rew   {cfg.use_indv_task_rew}",
        ],
    )

    buffer = cfg.n_steps * cfg.n_train_envs
    section(
        "ppo",
        [
            f"device              {cfg.device}",
            f"lr                  {lr_desc(cfg.learning_rate)}",
            f"n_steps x n_envs    {cfg.n_steps} x {cfg.n_train_envs} = {buffer:,}",
            f"batch_size          {cfg.batch_size}  ({buffer % cfg.batch_size} remainder)",
            f"n_epochs            {cfg.n_epochs}",
            f"gamma / lambda      {cfg.gamma} / {cfg.gae_lambda}",
            f"clip_range          {cfg.clip_range}",
            f"vf_coef / ent_coef  {cfg.vf_coef} / {cfg.ent_coef}",
            f"max_grad_norm       {cfg.max_grad_norm}",
            f"collect_data        {model.collect_data}",
        ],
    )

    # extract layer sizes from mlp_extractor
    def net_summary(net):
        sizes = [str(l.out_features) for l in net if hasattr(l, "out_features")]
        return " -> ".join(sizes)

    actor = net_summary(p.mlp_extractor.policy_net)
    critic = net_summary(p.mlp_extractor.value_net)
    act_out = p.action_net.out_features
    std = (
        f"{torch.exp(p.log_std).mean().item():.3f}" if hasattr(p, "log_std") else "n/a"
    )
    section(
        "network",
        [
            f"actor  (frozen)  in -> {actor} -> {act_out}",
            f"critic (train)   in -> {critic} -> 1",
            f"activation          {cfg.activation_fn.__name__}",
            f"action std (init)   {std}",
        ],
    )

    section(
        "warm-start",
        [
            f"load_actor_from     {cfg.load_actor_from}",
            f"init_log_std        {cfg.init_log_std}",
        ],
    )

    print(f"\n{'=' * 44}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain a PPO_BC critic from a named preset (actor frozen)."
    )
    parser.add_argument(
        "preset",
        choices=sorted(PRESETS.keys()),
        help="which pretrain_critic_config preset to run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PRESETS[args.preset]
    print(f"Using preset: {args.preset}  ->  experiment {cfg.experiment_name}")
    main(cfg)
