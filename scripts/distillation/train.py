# https://arxiv.org/pdf/2505.11164

import os
import warnings
import time
import subprocess
import threading
from datetime import datetime

from gymnasium import make
from stable_baselines3 import PPO

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pygame
from pynput import keyboard as kb
from tqdm import tqdm
from utils.paths import MODELS_DIR, LOGS_DIR, ROOT
from utils.logging import fmt_duration
from wrappers.bipedal_walker.distill_env import DistillEnv
from mdp.bipedal_walker.student import (
    StudentModelMLV2,
    OBS_SIZE_V2,
    ACT_SIZE,
)


EXPERT_MODEL_PATHS = [
    "experts/walk_forward",
    "experts/walk_backward",
    "experts/hop_forward",
    "experts/body_tilt",
]

TASK_NAMES = [
    "walk_forward",
    "walk_backward",
    "flamingo",
    "tilt",
]

MODEL_SIZE    = "ml"
MAJOR_VERSION = 1
MINOR_VERSION = 1
NOISE_COEF    = "n00"
MIX_MODE      = "mix"  # "mix" or "nomix"

MODEL_NAME      = f"{MODEL_SIZE}.{MAJOR_VERSION}.{MINOR_VERSION}.{NOISE_COEF}.{MIX_MODE}"
EXPERIMENT_NAME = f"distill_v2/{MODEL_NAME}"
# + datetime.today().strftime("-%H_%M_%S-%Y_%m_%d")
MIX_IRRELEVANT_INPUT    = MIX_MODE == "mix"
ADVERSARIAL_TASK_SELECT = True

_sim_paused = False
_sim_step = False
_sim_res = False


def main():
    global _sim_paused, _sim_step, _sim_res

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type  # type: ignore
    else:
        device = "cpu"

    print("Loading environments...")

    # 0: walk forward
    # 1: walk backward
    # 2: flamingo
    # 3: tilt

    env = DistillEnv(make("BipedalWalker-v3", render_mode=None), ep_time=7)
    eval_env = DistillEnv(make("BipedalWalker-v3", render_mode=None), ep_time=10)

    # env.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=10, interp_time=1, zero_prob=0.15)
    # eval_env.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=10, interp_time=1, zero_prob=0.15)

    # load models
    print("Loading experts...")
    EXPERT_MODELS = [
        PPO.load(MODELS_DIR / i, env=None, device="cpu") for i in EXPERT_MODEL_PATHS
    ]
    print("Loading student...")
    student = StudentModelMLV2()

    # helpers for training and eval
    def configureEnv(e: DistillEnv, task_id: int):
        # config the env to a certain task
        if task_id == 0:  # walk forward
            x_range = (0.0, 40.0)
            vel_range = (0.0, 5.0)
            e.config_hull_reset(x_range=x_range)
            e.config_cmd_vel(sample_range=vel_range, interp_time=0.5, switch_time=3, zero_prob=0.2)
            
            if MIX_IRRELEVANT_INPUT:  # mix in random tilt commands as well
                e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
            else:  # reset tilt command to 0 for clean input
                e.config_cmd_tilt(zero_prob=1)
        elif task_id == 1:  # walk backward
            x_range = (40.0, 80.0)
            vel_range = (-5.0, 0.0)
            e.config_hull_reset(x_range=x_range)
            e.config_cmd_vel(sample_range=vel_range, interp_time=0.5, switch_time=3, zero_prob=0.2)
            
            if MIX_IRRELEVANT_INPUT:  # mix in random tilt commands as well
                e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
            else:  # reset tilt command to 0 for clean input
                e.config_cmd_tilt(zero_prob=1)
        elif task_id == 2:  # flamingo
            x_range = (20.0, 60.0)
            e.config_hull_reset(x_range=x_range)
            
            if MIX_IRRELEVANT_INPUT:  # mix in random tilt and velocity commands
                e.config_cmd_vel(sample_range=(-5.0, 5.0), switch_time=3, interp_time=0.5, zero_prob=0.2)
                e.config_cmd_tilt(sample_range=(-0.75, 0.75), switch_time=3, interp_time=0.5, zero_prob=0.15)
            else:  # reset tilt and velocity command to 0 for clean input
                e.config_cmd_vel(zero_prob=1)
                e.config_cmd_tilt(zero_prob=1)
        elif task_id == 3:  # tilt
            x_range = (20.0, 60.0)
            tilt_range = (-0.75, 0.75)
            e.config_hull_reset(x_range=x_range)
            e.config_cmd_tilt(sample_range=tilt_range, switch_time=3, interp_time=0.5, zero_prob=0.15)
            
            if MIX_IRRELEVANT_INPUT:  # mix in random velocity commands
                e.config_cmd_vel(sample_range=(-5.0, 5.0), switch_time=3, interp_time=0.5, zero_prob=0.2)
            else:  # reset velocity command to 0 for clean input
                e.config_cmd_vel(zero_prob=1)

    def forwardExpert(obs: np.ndarray, task_id: int, cmd_vel: float, cmd_tilt: float = 0.0) -> np.ndarray:
        # body_tilt expert expects [proprio, cmd_tilt]
        if task_id == 3:
            action, _ = EXPERT_MODELS[3].predict(np.append(obs, cmd_tilt), deterministic=True)
            return action
        # for flamingo, poll the expert with a cmd velocity of 0
        if task_id == 2:
            action, _ = EXPERT_MODELS[2].predict(np.append(obs, 0), deterministic=True)
            return action

        # at 0 cmd velocity, default locomotion tasks to forward ones
        if cmd_vel == 0 and task_id == 1:
            task_id = 0

        action, _ = EXPERT_MODELS[task_id].predict(np.append(obs, cmd_vel), deterministic=True)
        return action

    def getTaskPMF(t: list[float], k: float):
        if ADVERSARIAL_TASK_SELECT:
            assert 0 <= k <= 1
            w = [max(t) - i for i in t]
            sum_w = sum(w)
            U = [1 / len(t)] * len(t)
            P = [U[i] if sum_w == 0 else w[i] / sum_w for i in range(len(t))]
            return [k * p_i + (1 - k) * u_i for p_i, u_i in zip(P, U)]
        else:
            return [1/len(t)] * len(t)

    # DAgger hyperparams
    T = 1500  # env steps per iter
    N = 80  # num iterations to go thru
    N_ACTIVE = N  # num iterations to sample from (disabled for now, don't need it)
    EPOCH = 30  # training epochs per iteration
    BATCH_SIZE = 256
    LR = 1e-3
    DECAY = 1e-2
    SCHED_RESTART_ITERS = 2  # dagger iterations per cosine restart
    ACT_VAR = 0.5  # action variance during data collection
    K = 0.85  # how much to prioritize choosing worst task (1 = max, 0 = uniform)
    T_EVAL = 2000  # eval steps per expert task

    # training settings
    CKPT_INT = 1  # how many iter before saving a checkpoint
    BEST_INT = 1  # how many iter before saving a best

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=DECAY)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=SCHED_RESTART_ITERS * EPOCH, eta_min=5e-4)

    D: list[tuple[np.ndarray, np.ndarray, int]] = []

    writer = SummaryWriter(log_dir=str(LOGS_DIR / EXPERIMENT_NAME))
    print_run_info(
        student,
        OBS_SIZE_V2,
        ACT_SIZE,
        device,
        N,
        T,
        EPOCH,
        BATCH_SIZE,
        LR,
        DECAY,
        EXPERIMENT_NAME,
    )

    # eval routine
    def evaluate(step: int):
        student.eval()
        student.to("cpu")
        task_losses = []
        all_time_alive = []
        with torch.no_grad():
            for task_id, task_name in enumerate(TASK_NAMES):
                configureEnv(eval_env, task_id)
                obs, info = eval_env.reset()
                cmd_vel = info["cmd"]["x_vel"]
                cmd_tilt = info["cmd"]["tilt"]
                done = False
                step_losses = []
                time_alive = []
                alive = 0

                for _ in range(T_EVAL):
                    if done:
                        obs, info = eval_env.reset()
                        cmd_vel = info["cmd"]["x_vel"]
                        cmd_tilt = info["cmd"]["tilt"]
                        done = False
                        time_alive.append(alive)
                        alive = 0
                    else:
                        alive += 1

                    act_expert = forwardExpert(obs, task_id, cmd_vel, cmd_tilt)
                    obs_s = StudentModelMLV2.obs(obs, task_id, cmd_vel, cmd_tilt)
                    pred = student(torch.tensor(obs_s, dtype=torch.float32))
                    target = torch.tensor(act_expert, dtype=torch.float32)

                    step_losses.append(F.mse_loss(pred, target).item())

                    obs, _, term, trunc, info = eval_env.step(pred.numpy())
                    cmd_vel = info["cmd"]["x_vel"]
                    cmd_tilt = info["cmd"]["tilt"]
                    done = term or trunc

                time_alive.append(alive)
                task_loss = float(np.mean(step_losses))
                task_time_alive = float(np.mean(time_alive))
                task_losses.append(task_loss)
                all_time_alive.append(task_time_alive)

                writer.add_scalar(f"eval/loss_{task_name}", task_loss, step)
                writer.add_scalar(f"eval/time_alive_{task_name}", task_time_alive, step)

        loss_total = float(np.mean(task_losses))
        avg_time_alive = float(np.mean(all_time_alive))

        writer.add_scalar("eval/loss_total", loss_total, step)
        writer.add_scalar("eval/avg_time_alive", avg_time_alive, step)

        return (loss_total, avg_time_alive, all_time_alive)

    # create all the necessary folders
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    if not os.path.exists(LOGS_DIR / "distill"):
        os.makedirs(LOGS_DIR / "distill")

    os.makedirs(MODELS_DIR / EXPERIMENT_NAME, exist_ok=True)

    # DAgger
    start_time = time.time()
    bar = tqdm(total=(N * T) + (N * EPOCH), desc="Training", ascii=" ░▒█")
    best_loss = float("inf")
    task_live_time = [1.0, 1.0, 1.0, 1.0]

    for n in range(N):
        iter_start = time.time()
        Di: list[tuple[np.ndarray, np.ndarray, int]] = []

        # 1. collect trajectories under current student policy
        student.to("cpu")
        student.eval()
        time_alive = []
        with torch.no_grad():
            done = True
            alive = 0

            for _ in range(T):
                if done:
                    current_task = int(
                        np.random.choice(4, p=getTaskPMF(task_live_time, K))
                    )
                    configureEnv(env, current_task)
                    obs, info = env.reset()
                    cmd_vel = info["cmd"]["x_vel"]
                    cmd_tilt = info["cmd"]["tilt"]
                    alive = 0

                act_expert = forwardExpert(obs, current_task, cmd_vel, cmd_tilt)
                # add zero-mean gaussian noise
                act_expert += np.random.normal(0, ACT_VAR**0.5, act_expert.shape)

                obs_s = StudentModelMLV2.obs(obs, current_task, cmd_vel, cmd_tilt)
                act_student = student(torch.tensor(obs_s, dtype=torch.float32)).numpy()

                alive += 1
                obs, _, term, trunc, info = env.step(act_student)
                cmd_vel = info["cmd"]["x_vel"]
                cmd_tilt = info["cmd"]["tilt"]
                done = term or trunc

                if done:
                    time_alive.append(alive)

                Di.append((obs_s, act_expert, current_task))
                bar.update(1)

        # 2. aggregate dataset
        D += Di

        # 3. build training batch, prioritising recent data
        obs_list, act_list, task_ids_all = zip(*D)  # type: ignore
        if len(D) > N_ACTIVE * T:
            # linearly increasing weights: oldest sample is around 0, newest = highest
            w = np.arange(1, len(D) + 1, dtype=np.float64)
            w /= w.sum()
            idx = np.random.choice(len(D), size=N_ACTIVE * T, replace=False, p=w)
            obs_arr = np.array(obs_list)[idx]
            act_arr = np.array(act_list)[idx]
            task_ids_all = tuple(np.array(task_ids_all)[idx])
        else:
            obs_arr = np.array(obs_list)
            act_arr = np.array(act_list)

        x_full = torch.tensor(obs_arr, dtype=torch.float32)
        y_full = torch.tensor(act_arr, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(x_full, y_full), batch_size=BATCH_SIZE, shuffle=True
        )

        student.to(device)
        student.train()

        for epoch in range(EPOCH):
            epoch_loss = 0.0
            for obs_batch, act_batch in loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)

                pred = student(obs_batch)
                loss = loss_fn(pred, act_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            writer.add_scalar("train/loss", epoch_loss / len(loader), n * EPOCH + epoch)
            # scheduler.step()
            bar.update(1)

        # log per-iteration scalars
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], n * EPOCH)
        writer.add_scalar("train/dataset_size", len(D), n * EPOCH)
        if time_alive:
            writer.add_scalar(
                "train/avg_time_alive", float(np.mean(time_alive)), n * EPOCH
            )

        # per-expert training loss on full accumulated dataset
        student.eval()
        student.to("cpu")
        with torch.no_grad():
            pred_all = student(x_full)
            total_D = len(task_ids_all)
            for task_id, task_name in enumerate(TASK_NAMES):
                indices = [i for i, t in enumerate(task_ids_all) if t == task_id]
                if indices:
                    idx_t = torch.tensor(indices)
                    writer.add_scalar(
                        f"train/loss_{task_name}",
                        F.mse_loss(pred_all[idx_t], y_full[idx_t]).item(),
                        n * EPOCH,
                    )
                writer.add_scalar(
                    f"train/task_pct_{task_name}",
                    len(indices) / total_D,
                    n * EPOCH,
                )

        # 4. eval
        eval_loss, _, task_live_time = evaluate(n * EPOCH)

        # 5. save checkpoints
        if n % CKPT_INT == 0 or n == N - 1:
            torch.save(
                {
                    "policy": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                str(MODELS_DIR / EXPERIMENT_NAME / f"distill_{n}.pt"),
            )
        if n % BEST_INT == 0:
            if eval_loss < best_loss:  # save best model
                torch.save(
                    {
                        "policy": student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    str(MODELS_DIR / EXPERIMENT_NAME / f"best.pt"),
                )
                best_loss = eval_loss

        iter_time = time.time() - iter_start
        iter_per_s = 1.0 / iter_time
        writer.add_scalar("train/iter_per_s", iter_per_s, n)
        writer.add_scalar("train/iter_time_s", iter_time, n)
        bar.set_postfix(iter_s=f"{iter_per_s:.3f}")

    bar.close()
    writer.close()

    duration = fmt_duration(time.time() - start_time)
    print(f"\nDone! Total time: {duration}")

    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "Finished in {duration}" with title "Distillation complete" subtitle "{EXPERIMENT_NAME}"',
            ],
            check=False,
        )
    except FileNotFoundError:
        pass
    try:
        play_sound(ROOT / "assets" / "train_finish.mp3")
    except Exception as e:
        print(f"(skipping play_sound: {e})")


def print_run_info(
    student, obs_size, act_size, device, N, T, EPOCH, BATCH_SIZE, LR, DECAY, run_name
):
    def section(title, lines):
        print(f"\n  {title}")
        print(f"  {'-' * 44}")
        for line in lines:
            print(f"    {line}")

    print(f"\n{'=' * 44}")
    print(f"  DAgger distillation  {run_name}")
    print(f"{'=' * 44}")

    section(
        "algorithm",
        [
            f"N (dagger iters)   {N}",
            f"T (steps / iter)   {T}",
            f"epochs / update    {EPOCH}",
            f"batch size         {BATCH_SIZE}",
            f"lr                 {LR}",
            f"weight decay       {DECAY}",
            f"total env steps    {N * T:,}",
        ],
    )

    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    section(
        "student network",
        [
            f"obs size   {obs_size}",
            f"act size   {act_size}",
            f"params     {n_params:,}",
            f"device     {device}",
        ]
        + [f"layer      {m}" for m in student.policy if hasattr(m, "in_features")],
    )

    section(
        "logging",
        [f"tensorboard  logs/{run_name}"],
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
    warnings.filterwarnings(
        "ignore",
        message="__array_wrap__ must accept context and return_scalar arguments",
        category=DeprecationWarning,
    )
    main()
