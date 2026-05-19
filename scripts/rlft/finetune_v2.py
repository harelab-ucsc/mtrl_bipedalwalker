from typing import OrderedDict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
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
from mdp.bipedal_walker.rlft_policy import RlFTPolicy, _MODEL_CONFIGS
from wrappers.bipedal_walker.rltf_env_v2 import RlFTEnvV2

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# =========================================

# distill model / experiment ident
MODEL_SIZE = "ml"
D_MAJOR = 1
D_MINOR = 1
D_NOISE_COEF = "n05"
D_MIX_MODE = "nomix"  # "mix" or "nomix"

# finetune model / experiment ident
CRITIC_SIZE   = "xxxl"
FT_MAJOR = 1
FT_MINOR = 0
GAMMA  = 0.975
LAMBDA = 0.95

D_MODEL_NAME = f"{MODEL_SIZE}.{D_MAJOR}.{D_MINOR}.{D_NOISE_COEF}.{D_MIX_MODE}"
FT_MODEL_NAME = f"{MODEL_SIZE}.{D_MAJOR}.{D_MINOR}.{D_NOISE_COEF}_{FT_MAJOR}.{FT_MINOR}.g{int(GAMMA * 100):02d}"

# Run controls
PRETRAIN_FROM_SCRATCH = True
RUN_FINETUNE = False

# paths
DISTILLED_STUDENT        = f"distill_v2/{D_MODEL_NAME}/best.pt"
PRETRAIN_EXPERIMENT_NAME = f"rlft/pretrain_v2/{D_MODEL_NAME}"
FINETUNE_EXPERIMENT_NAME = (
    f"rlft/finetuned/{D_MODEL_NAME}"
    f"-{datetime.today().strftime('%H_%M_%S-%Y_%m_%d')}"
)

# Used when PRETRAIN_FROM_SCRATCH = False:
PRETRAINED_MODEL_PATH = MODELS_DIR / f"rlft/pretrain_v2/{FT_MODEL_NAME}/best_model"

# --- environment ---
N_TRAIN_ENVS        = 14
N_EVAL_ENVS         = 5
EP_TIME             = 10
VEL_SAMPLE_RANGE    = (-5, 5)
TILT_SAMPLE_RANGE   = (-0.75, 0.75)
VEL_SAMPLE_ZERO     = 0.15
VEL_SWITCHING_FREQ  = 2
TILT_SWITCHING_FREQ = 3
TASK_SWITCHING_FREQ = 5
VEL_INTERP_SPEED    = 10.0

# --- pretraining hyperparams ---
PRETRAIN_TIMESTEPS    = 200 * 1024 * N_TRAIN_ENVS
PRETRAIN_LR           = 1e-3
PRETRAIN_N_EPOCHS     = 10
PRETRAIN_N_STEPS      = 1024
PRETRAIN_BATCH_SIZE   = 256
PRETRAIN_ENT_COEF     = 0.0
PRETRAIN_VF_COEF      = 1.0
PRETRAIN_INIT_LOG_STD = np.log(0.1)

# --- finetuning hyperparams ---
FINETUNE_TIMESTEPS      = 200 * 1024 * N_TRAIN_ENVS
FINETUNE_LR_START       = 2e-5
FINETUNE_LR_END         = 8e-6
FINETUNE_LR_FRACTION    = 0.5
FINETUNE_N_EPOCHS       = 25
FINETUNE_N_STEPS        = 1024
FINETUNE_BATCH_SIZE     = 64
FINETUNE_ENT_COEF       = 0.006
FINETUNE_CLIP_RANGE     = 0.1
FINETUNE_INIT_LOG_STD   = np.log(1.0)
FINETUNE_ADVERSARIAL    = False  # whether to pick adversarial command inputs or not

# =========================================


def make_env():
    env = gym.make("BipedalWalker-v3")
    env = Monitor(
        RlFTEnvV2(
            env,
            ep_time=EP_TIME,
            vel_sample_range=VEL_SAMPLE_RANGE,
            vel_sample_zero=VEL_SAMPLE_ZERO,
            vel_switching_freq=VEL_SWITCHING_FREQ,
            vel_interp_speed=VEL_INTERP_SPEED,
            task_switching_freq=TASK_SWITCHING_FREQ,
        )
    )
    return env


def main():
    assert PRETRAIN_FROM_SCRATCH or RUN_FINETUNE, "nothing to do"

    t0 = time.time()

    if PRETRAIN_FROM_SCRATCH:
        pretrained_path = run_pretraining(t0)
    else:
        pretrained_path = PRETRAINED_MODEL_PATH

    if RUN_FINETUNE:
        run_finetuning(pretrained_path, t0)


def run_pretraining(t0: float) -> Path:
    print(f"[{(time.time() - t0):.2f}s] Loading environments...")
    train_env = SubprocVecEnv([make_env for _ in range(N_TRAIN_ENVS)])
    eval_env  = SubprocVecEnv([make_env for _ in range(N_EVAL_ENVS)])

    print(f"[{(time.time() - t0):.2f}s] Creating model...")
    hidden_dims        = _MODEL_CONFIGS[MODEL_SIZE]
    critic_hidden_dims = _MODEL_CONFIGS[CRITIC_SIZE]
    model = PPO(
        RlFTPolicy,
        env=train_env,
        policy_kwargs=dict(hidden_dims=hidden_dims, critic_hidden_dims=critic_hidden_dims, activation_fn=torch.nn.ELU),
        learning_rate=PRETRAIN_LR,
        n_epochs=PRETRAIN_N_EPOCHS,
        n_steps=PRETRAIN_N_STEPS,
        batch_size=PRETRAIN_BATCH_SIZE,
        ent_coef=PRETRAIN_ENT_COEF,
        vf_coef=PRETRAIN_VF_COEF,
        clip_range=FINETUNE_CLIP_RANGE,
        gamma=GAMMA,
        gae_lambda=LAMBDA,
    )

    print(f"[{(time.time() - t0):.2f}s] Preloading policy...")
    student_path = MODELS_DIR / DISTILLED_STUDENT
    student_sd: OrderedDict = torch.load(
        student_path, map_location="cpu", weights_only=False
    )["policy"]
    layer_idx = [int(i.split(".")[1]) for i in list(student_sd.keys())[::2]]

    sb3_policy = model.policy
    mlp_ext    = sb3_policy.mlp_extractor
    action_net = sb3_policy.action_net

    with torch.no_grad():
        for idx in layer_idx[:-1]:
            mlp_ext.policy_net[idx].weight.copy_(student_sd[f"policy.{idx}.weight"])  # type: ignore
            mlp_ext.policy_net[idx].bias.copy_(student_sd[f"policy.{idx}.bias"])  # type: ignore
        action_net.weight.copy_(student_sd[f"policy.{layer_idx[-1]}.weight"])  # type: ignore
        action_net.bias.copy_(student_sd[f"policy.{layer_idx[-1]}.bias"])  # type: ignore

    for p in mlp_ext.policy_net.parameters():
        p.requires_grad_(False)
    for p in action_net.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        model.policy.log_std.fill_(PRETRAIN_INIT_LOG_STD)
    sb3_policy.log_std.requires_grad_(False)

    print(f"[{(time.time() - t0):.2f}s] Configuring logger...")
    model.set_logger(configure(str(LOGS_DIR / PRETRAIN_EXPERIMENT_NAME), ["tensorboard"]))
    train_env.reset()

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODELS_DIR}/{PRETRAIN_EXPERIMENT_NAME}",
        eval_freq=max(50000 // N_TRAIN_ENVS, 1),
        n_eval_episodes=5,
        verbose=0,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(500000 // N_TRAIN_ENVS, 1),
        save_path=f"{MODELS_DIR}/{PRETRAIN_EXPERIMENT_NAME}/",
    )

    print_run_info(train_env, model, PRETRAIN_EXPERIMENT_NAME, phase="pretraining")

    print(f"[{(time.time() - t0):.2f}s] Starting pretraining ({PRETRAIN_TIMESTEPS:,} timesteps)...")
    start = time.time()
    model.learn(
        total_timesteps=PRETRAIN_TIMESTEPS,
        callback=CallbackList([StandardTBCallback(), RewardTermLogger(), eval_cb, ckpt_cb]),
        progress_bar=True,
    )

    duration = fmt_duration(time.time() - start)
    print(f"[{(time.time() - t0):.2f}s] Pretraining done. Total time: {duration}")
    subprocess.run(
        [
            "osascript", "-e",
            f'display notification "Finished in {duration}" with title "Pretraining complete" subtitle "{PRETRAIN_EXPERIMENT_NAME}"',
        ],
        check=False,
    )

    print(f"[{(time.time() - t0):.2f}s] Cleaning up pretraining resources...")
    train_env.close()
    eval_env.close()
    del model, train_env, eval_env

    return MODELS_DIR / f"{PRETRAIN_EXPERIMENT_NAME}" / "best_model.zip"


def run_finetuning(pretrained_path: Path, t0: float) -> None:
    print(f"[{(time.time() - t0):.2f}s] Loading environments...")
    train_env = SubprocVecEnv([make_env for _ in range(N_TRAIN_ENVS)])
    eval_env  = SubprocVecEnv([make_env for _ in range(N_EVAL_ENVS)])

    print(f"[{(time.time() - t0):.2f}s] Loading pretrained model from {pretrained_path}...")
    model = PPO.load(
        pretrained_path,
        env=train_env,
        custom_objects={
            "learning_rate": LinearSchedule(FINETUNE_LR_START, FINETUNE_LR_END, FINETUNE_LR_FRACTION),
            "n_epochs":      FINETUNE_N_EPOCHS,
            "n_steps":       FINETUNE_N_STEPS,
            "batch_size":    FINETUNE_BATCH_SIZE,
            "ent_coef":      FINETUNE_ENT_COEF,
            "gamma":         GAMMA,
            "gae_lambda":    LAMBDA,
        },
    )

    print(f"[{(time.time() - t0):.2f}s] Reinitializing log_std...")
    with torch.no_grad():
        model.policy.log_std.fill_(FINETUNE_INIT_LOG_STD)

    print(f"[{(time.time() - t0):.2f}s] Configuring logger...")
    model.set_logger(configure(str(LOGS_DIR / FINETUNE_EXPERIMENT_NAME), ["tensorboard"]))
    train_env.reset()

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODELS_DIR}/{FINETUNE_EXPERIMENT_NAME}/best",
        eval_freq=max(50000 // N_TRAIN_ENVS, 1),
        n_eval_episodes=5,
        verbose=0,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(500000 // N_TRAIN_ENVS, 1),
        save_path=f"{MODELS_DIR}/{FINETUNE_EXPERIMENT_NAME}/",
    )

    print_run_info(train_env, model, FINETUNE_EXPERIMENT_NAME, phase="finetuning")

    print(f"[{(time.time() - t0):.2f}s] Starting finetuning ({FINETUNE_TIMESTEPS:,} timesteps)...")
    start = time.time()
    model.learn(
        total_timesteps=FINETUNE_TIMESTEPS,
        reset_num_timesteps=True,
        callback=CallbackList([StandardTBCallback(), RewardTermLogger(), eval_cb, ckpt_cb]),
        progress_bar=True,
    )

    duration = fmt_duration(time.time() - start)
    print(f"[{(time.time() - t0):.2f}s] Finetuning done. Total time: {duration}")
    print(f"Experiment name: {FINETUNE_EXPERIMENT_NAME}")
    subprocess.run(
        [
            "osascript", "-e",
            f'display notification "Finished in {duration}" with title "Finetuning complete" subtitle "{FINETUNE_EXPERIMENT_NAME}"',
        ],
        check=False,
    )
    play_sound(ROOT / "assets" / "train_finish.mp3")

    train_env.close()
    eval_env.close()


def print_run_info(env, model, experiment_name: str, phase: str) -> None:
    env_id = env.get_attr("spec")[0].id
    obs = env.observation_space
    act = env.action_space
    p   = model.policy

    def section(title, lines):
        print(f"\n  {title}")
        print(f"  {'-' * 40}")
        for line in lines:
            print(f"    {line}")

    print(f"\n{'=' * 44}")
    print(f"  {phase:<20} {experiment_name}")
    if phase == "pretraining":
        print(f"  frozen policy        {DISTILLED_STUDENT}")
        print(f"  Note: actor is frozen; only the value network trains.")
    else:
        print(f"  pretrained model     {PRETRAINED_MODEL_PATH if not PRETRAIN_FROM_SCRATCH else f'{PRETRAIN_EXPERIMENT_NAME}'}")
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
            f"clip_range        {model.clip_range}",
            f"target_kl         {model.target_kl}",
            f"entropy_coef      {model.ent_coef}",
            f"stats_win_size    {model._stats_window_size}",
            f"lr                {model.learning_rate}",
            f"seed              {model.seed}",
        ],
    )

    def net_summary(net):
        return " -> ".join(str(l.out_features) for l in net if hasattr(l, "out_features"))

    actor   = net_summary(p.mlp_extractor.policy_net)
    critic  = net_summary(p.mlp_extractor.value_net)
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
