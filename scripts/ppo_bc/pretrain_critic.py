"""
Critic pretraining for PPO_BC.

Loads a trained PPO_BC actor, drops a fresh critic next to it, and runs PPO
with the actor frozen so only the value network learns. Output is a PPO_BC
zip that train.py can resume via LOAD_MODEL
"""

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

from datetime import datetime
import time
import os
import subprocess

from utils.paths import MODELS_DIR, LOGS_DIR
from utils.logging import StandardTBCallback, RewardTermLogger, fmt_duration

from wrappers.ppo_bc.ppo_bc_env import RlFTEnv
from ppo_bc_sb3 import PPO_BC, PpoBcPolicy, load_expert

if not os.path.exists(MODELS_DIR / "ppo_bc"):
    os.makedirs(MODELS_DIR / "ppo_bc")
if not os.path.exists(LOGS_DIR / "ppo_bc"):
    os.makedirs(LOGS_DIR / "ppo_bc")

# =========================================

# output
EXPERIMENT_NAME = "ppo_bc/critic_pretrain/1.2.0"
TIMESTEPS = 200 * 1024 * 14

# ppo bc prior
LOAD_ACTOR_FROM = MODELS_DIR / "ppo_bc" / "1.2.0-21_04_31-2026_05_26" / "rl_model_2799664_steps.zip"

# environment params
N_TRAIN_ENVS         = 14
EP_TIME              = 10
CMD_SWITCHING_TIME   = (3.0, 4.0)   # (vel, tilt)
TASK_SWITCHING_TIME  = 6.0
CMD_INTERP_SPEED     = (5.0, 1.0)
CMD_SAMPLE_RANGE     = ((-5.0, 5.0), (-0.75, 0.75))
CMD_SAMPLE_ZERO      = (0.2, 0.15)
# needed for task combination
ALLOWED_TASK_MIXING  = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
]
HULL_X_RANGE         = (20.0, 60.0)

# pretraining hyperparams (high vf_coef, no ent bonus, no BC)
PRETRAIN_LR           = 1e-3
PRETRAIN_N_EPOCHS     = 20
PRETRAIN_N_STEPS      = 1024
PRETRAIN_BATCH_SIZE   = 256
PRETRAIN_ENT_COEF     = 0.0
PRETRAIN_VF_COEF      = 1.0
PRETRAIN_INIT_LOG_STD = float(np.log(0.1))   # tight policy — stay near actor's mode

# shared PPO config
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_RANGE     = 0.2
MAX_GRAD_NORM  = 0.5
DEVICE         = torch.device("cpu")

# network arch
# HIDDEN_DIMS must match the actor in LOAD_ACTOR_FROM
HIDDEN_DIMS          = [256, 128, 64]
CRITIC_HIDDEN_DIMS   = [1024, 512, 512, 256, 256]
ACTIVATION_FN        = torch.nn.ELU

# task / DAgger bits (ctor needs them; relabeling is disabled below)
TASK_BITS         = 3
N_PROPRIO         = 14
ACT_VAR_FLOOR     = 0.0

# Expert checkpoints. Required by PPO_BC ctor but never invoked (collect_data=False).
EXPERT_PATHS = {
    "walk_forward":  MODELS_DIR / "experts" / "walk_forward",
    "walk_backward": MODELS_DIR / "experts" / "walk_backward",
    "hop_forward":   MODELS_DIR / "experts" / "hop_forward",
    "body_tilt":     MODELS_DIR / "experts" / "body_tilt",
}

# =========================================


def build_experts():
    """Duplicated from train.py. Required by PPO_BC ctor; not actually polled
    when collect_data=False, but the dict needs to be present and non-empty."""
    walk_fwd = load_expert(EXPERT_PATHS["walk_forward"])
    walk_bwd = load_expert(EXPERT_PATHS["walk_backward"])
    hop_fwd  = load_expert(EXPERT_PATHS["hop_forward"])
    tilt     = load_expert(EXPERT_PATHS["body_tilt"])

    def walk_call(obs):
        cmd_vel = obs[:, N_PROPRIO:N_PROPRIO + 1]
        x = np.concatenate([obs[:, :N_PROPRIO], cmd_vel], axis=-1)
        mask_fwd = cmd_vel[:, 0] >= 0
        act = np.zeros((obs.shape[0], 4), dtype=np.float32)
        if mask_fwd.any():
            act[mask_fwd] = walk_fwd.predict(x[mask_fwd], deterministic=True)[0]
        if (~mask_fwd).any():
            act[~mask_fwd] = walk_bwd.predict(x[~mask_fwd], deterministic=True)[0]
        return act

    def flamingo_call(obs):
        zero = np.zeros((obs.shape[0], 1), dtype=obs.dtype)
        x = np.concatenate([obs[:, :N_PROPRIO], zero], axis=-1)
        return hop_fwd.predict(x, deterministic=True)[0]

    def tilt_call(obs):
        cmd_tilt = obs[:, N_PROPRIO + 1:N_PROPRIO + 2]
        x = np.concatenate([obs[:, :N_PROPRIO], cmd_tilt], axis=-1)
        return tilt.predict(x, deterministic=True)[0]

    return {
        (1, 0, 0): walk_call,
        (0, 1, 0): flamingo_call,
        (0, 0, 1): tilt_call,
    }


def make_env():
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
            use_rew_for_individual_tasks=True,
            hull_x_range=HULL_X_RANGE,
        )
    )
    return env


def load_actor_only(model: PPO_BC, zip_path) -> None:
    """Copy actor weights (policy_net trunk + action_net + log_std) from a
    saved PPO_BC zip into `model`, leaving the critic at its random init.

    PPO_BC.load reconstructs the source policy using *its own* saved
    policy_kwargs, so the source critic dims don't have to match the
    destination's CRITIC_HIDDEN_DIMS. The actor dims do have to match — we
    assert layer-by-layer.
    """
    src = PPO_BC.load(
        zip_path,
        experts=model.experts,
        task_bits=model.task_bits,
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
                f"vs src {tuple(src_sd[k].shape)} — HIDDEN_DIMS must match the actor zip."
            )
            dst_sd[k] = src_sd[k]
    model.policy.load_state_dict(dst_sd)


def main():
    assert LOAD_ACTOR_FROM.exists(), f"LOAD_ACTOR_FROM does not exist: {LOAD_ACTOR_FROM}"

    print("Loading environments...")
    train_env = SubprocVecEnv([make_env for _ in range(N_TRAIN_ENVS)])

    print("Loading experts (ctor requirement only; not polled)...")
    experts = build_experts()

    policy_kwargs = dict(
        hidden_dims=HIDDEN_DIMS,
        critic_hidden_dims=CRITIC_HIDDEN_DIMS,
        activation_fn=ACTIVATION_FN,
    )

    # Fresh PPO_BC. collect_data=False disables DAgger relabeling + dataset
    # growth entirely; bc_coef=0 makes the BC term explicitly inert.
    model = PPO_BC(
        PpoBcPolicy,
        train_env,
        experts=experts,
        task_bits=TASK_BITS,
        act_var_floor=ACT_VAR_FLOOR,
        bc_coef=0.0,
        collect_data=False,
        verbose=0,
        learning_rate=PRETRAIN_LR,
        n_epochs=PRETRAIN_N_EPOCHS,
        n_steps=PRETRAIN_N_STEPS,
        batch_size=PRETRAIN_BATCH_SIZE,
        ent_coef=PRETRAIN_ENT_COEF,
        vf_coef=PRETRAIN_VF_COEF,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        device=DEVICE,
    )

    print(f"Loading actor weights from {LOAD_ACTOR_FROM}...")
    load_actor_only(model, LOAD_ACTOR_FROM)

    # Freeze actor + action head + log_std so only the critic learns.
    for p in model.policy.mlp_extractor.policy_net.parameters():
        p.requires_grad_(False)
    for p in model.policy.action_net.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        model.policy.log_std.fill_(PRETRAIN_INIT_LOG_STD)
    model.policy.log_std.requires_grad_(False)

    model.set_logger(configure(str(LOGS_DIR / EXPERIMENT_NAME), ["tensorboard"]))
    train_env.reset()

    ckpt_cb = CheckpointCallback(
        save_freq=max(500000 // train_env.num_envs, 1),
        save_path=f"{MODELS_DIR}/{EXPERIMENT_NAME}/",
    )

    print_run_info(train_env, model, EXPERIMENT_NAME)

    print(f"Starting critic pretraining ({TIMESTEPS:,} timesteps)...")
    start = time.time()
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        callback=CallbackList([StandardTBCallback(), RewardTermLogger(), ckpt_cb]),
        progress_bar=True,
    )

    final_path = f"{MODELS_DIR}/{EXPERIMENT_NAME}/final.zip"
    model.save(final_path)
    print(f"Saved final model to {final_path}")

    duration = fmt_duration(time.time() - start)
    print(f"Done! Total time: {duration}")
    print(f"Experiment name: {EXPERIMENT_NAME}")

    try:
        subprocess.run(
            [
                "osascript", "-e",
                f'display notification "Finished in {duration}" with title "Critic pretrain complete" subtitle "{EXPERIMENT_NAME}"',
            ],
            check=False,
        )
    except FileNotFoundError:
        pass


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
    print(f"  critic_pretrain  {experiment_name}")
    print(f"  frozen actor     {LOAD_ACTOR_FROM}")
    print(f"  Note: actor is frozen; only the value network trains.")
    print(f"{'=' * 44}")

    section(
        "environment",
        [
            f"{env.num_envs}x  {env_id}",
            f"obs  {obs.shape}  {obs.dtype}",
            f"act  {act.shape}  [{act.low[0]:.1f}, {act.high[0]:.1f}]",
            f"task mixings     {ALLOWED_TASK_MIXING}",
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
            f"vf_coef           {model.vf_coef}",
            f"ent_coef          {model.ent_coef}",
            f"lr                {model.learning_rate}",
            f"init_log_std      {PRETRAIN_INIT_LOG_STD:.4f}",
            f"collect_data      {model.collect_data}",
        ],
    )

    def net_summary(net):
        return " -> ".join(str(l.out_features) for l in net if hasattr(l, "out_features"))

    actor = net_summary(p.mlp_extractor.policy_net)
    critic = net_summary(p.mlp_extractor.value_net)
    act_out = p.action_net.out_features
    section(
        "network",
        [
            f"actor  (frozen)  in -> {actor} -> {act_out}",
            f"critic (train)   in -> {critic} -> 1",
        ],
    )

    print(f"\n{'=' * 44}\n")


if __name__ == "__main__":
    main()
