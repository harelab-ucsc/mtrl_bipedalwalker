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
from torch.optim.lr_scheduler import CosineAnnealingLR

import pygame
from pynput import keyboard as kb
from tqdm import tqdm
from utils.paths import MODELS_DIR, LOGS_DIR, ROOT
from utils.logging import fmt_duration
from wrappers.bipedal_walker.distill_env import DistillEnv
from mdp.bipedal_walker.student import StudentModel, OBS_SIZE, ACT_SIZE
    
    
EXPERT_MODEL_PATHS = [
    "experts/walk_forward",
    "experts/walk_backward",
    "experts/hop_forward",
    "experts/hop_backward",
]

TASK_NAMES = ["walk_forward", "walk_backward", "hop_forward", "hop_backward"]

EXPERIMENT_NAME = "distill/temp_5" + datetime.today().strftime("-%H_%M_%S-%Y_%m_%d")

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
    # 2: hop forward
    # 3: hop backward
    
    env = DistillEnv(make("BipedalWalker-v3", render_mode=None), ep_time=7)
    eval_env = DistillEnv(make("BipedalWalker-v3", render_mode=None), ep_time=10)

    # load  models
    print("Loading experts...")
    EXPERT_MODELS = [
        PPO.load(MODELS_DIR / i, env=None, device="cpu") for i in EXPERT_MODEL_PATHS
    ]
    print("Loading student...")
    student = StudentModel()

    # helpers for training and eval
    def configureEnv(e: DistillEnv, task_id: int):
        # config the env to a certain task
        if task_id == 0:  # walk forward
            x_range = (0.0, 40.0)
            vel_range = (0.0, 5.0)
        elif task_id == 2:  # hop forward
            x_range = (0.0, 40.0)
            vel_range = (0.0, 5.0)
        elif task_id == 1:  # walk backward
            x_range = (40.0, 80.0)
            vel_range = (-5.0, 0.0)
        elif task_id == 3:  # hop backward
            x_range = (40.0, 80.0)
            vel_range = (-5.0, 0.0)
        e.config_hull_reset(x_range=x_range)
        e.config_cmd_vel(sample_range=vel_range, interp_time=0.5, switch_time=3, zero_prob=0.35)

    def forwardExpert(obs: np.ndarray, task_id: int, cmd_vel: float) -> np.ndarray:
        # at 0 cmd velocity, default locomotion tasks to forward ones
        if cmd_vel == 0:
            if task_id == 1:  # walk backwards
                task_id = 0
            elif task_id == 3:  # hop backwards
                task_id = 2
                
        # get an expert's opinion lol
        action, _ = EXPERT_MODELS[task_id].predict(np.append(obs, cmd_vel), deterministic=True)
        return action
    
    def getTaskPMF(t: list[float], k: float):
        assert 0 <= k <= 1
        w = [max(t) - i for i in t]
        sum_w = sum(w)
        U = [1/len(t)] * len(t)
        P = [U[i] if sum_w == 0 else w[i]/sum_w for i in range(len(t))]
        return [k*p_i + (1-k)*u_i for p_i, u_i in zip(P, U)]

    # DAgger hyperparams
    T = 1500                # env steps per iter
    N = 40                  # num iterations to go thru
    EPOCH = 30              # training epochs per iteration
    BATCH_SIZE = 256
    LR = 1e-3
    DECAY = 1e-2
    ACT_VAR = 0.2           # action variance during data collection
    K = 0.9                 # how much to prioritize choosing worst task (1 = max, 0 = uniform)
    T_EVAL = 500            # eval steps per expert task
    
    # training settings
    CKPT_INT = 1            # how many iter before saving a checkpoint
    BEST_INT = 1            # how many iter before saving a best

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=DECAY)
    # scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH*N, eta_min=3e-5)

    D: list[tuple[np.ndarray, np.ndarray, int]] = []

    writer = SummaryWriter(log_dir=str(LOGS_DIR / "distill" / EXPERIMENT_NAME))
    print_run_info(student, OBS_SIZE, ACT_SIZE, device, N, T, EPOCH, BATCH_SIZE, LR, DECAY, EXPERIMENT_NAME)

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
                done = False
                step_losses = []
                time_alive = []
                alive = 0

                for _ in range(T_EVAL):
                    if done:
                        obs, info = eval_env.reset()
                        cmd_vel = info["cmd"]["x_vel"]
                        done = False
                        time_alive.append(alive)
                        alive = 0
                    else:
                        alive += 1

                    act_expert = forwardExpert(obs, task_id, cmd_vel)
                    obs_s = StudentModel.obs(obs, task_id, cmd_vel)
                    pred = student(torch.tensor(obs_s, dtype=torch.float32))
                    target = torch.tensor(act_expert, dtype=torch.float32)

                    step_losses.append(F.mse_loss(pred, target).item())

                    obs, _, term, trunc, info = eval_env.step(pred.numpy())
                    cmd_vel = info["cmd"]["x_vel"]
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
    task_live_time = [1., 1., 1., 1.]

    for n in range(N):
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
                    current_task = int(np.random.choice(4, p=getTaskPMF(task_live_time, K)))
                    configureEnv(env, current_task)
                    obs, info = env.reset()
                    cmd_vel = info["cmd"]["x_vel"]
                    alive = 0

                act_expert = forwardExpert(obs, current_task, cmd_vel)
                # add zero-mean gaussian noise
                act_expert += np.random.normal(0, ACT_VAR**0.5, act_expert.shape)
                
                obs_s = StudentModel.obs(obs, current_task, cmd_vel)
                act_student = student(torch.tensor(obs_s, dtype=torch.float32)).numpy()

                alive += 1
                obs, _, term, trunc, info = env.step(act_student)
                cmd_vel = info["cmd"]["x_vel"]
                done = term or trunc

                if done:
                    time_alive.append(alive)

                Di.append((obs_s, act_expert, current_task))
                bar.update(1)
        
        # 2. aggregate dataset
        D += Di

        # 3. train student on full (D)ataset
        obs_list, act_list, task_ids_all = zip(*D)  # type: ignore
        x_full = torch.tensor(np.array(obs_list), dtype=torch.float32)
        y_full = torch.tensor(np.array(act_list), dtype=torch.float32)

        loader = DataLoader(TensorDataset(x_full, y_full), batch_size=BATCH_SIZE, shuffle=True)

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
                # scheduler.step()
                epoch_loss += loss.item()

            writer.add_scalar("train/loss", epoch_loss / len(loader), n * EPOCH + epoch)
            bar.update(1)

        # log per-iteration scalars
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], n * EPOCH)
        writer.add_scalar("train/dataset_size", len(D), n * EPOCH)
        if time_alive:
            writer.add_scalar("train/avg_time_alive", float(np.mean(time_alive)), n * EPOCH)

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
            torch.save({
                "policy": student.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, str(MODELS_DIR / EXPERIMENT_NAME / f"distill_{n}.pt"))
        if n % BEST_INT == 0:
            if eval_loss < best_loss:  # save best model
                torch.save({
                    "policy": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, str(MODELS_DIR / EXPERIMENT_NAME / f"best.pt"))
                best_loss = eval_loss

    bar.close()
    writer.close()

    duration = fmt_duration(time.time() - start_time)
    print(f"\nDone! Total time: {duration}")

    try:
        subprocess.run(
            [
                "osascript", "-e",
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


def print_run_info(student, obs_size, act_size, device, N, T, EPOCH, BATCH_SIZE, LR, DECAY, run_name):
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

    section(
        "student network",
        [
            f"obs size   {obs_size}",
            f"act size   {act_size}",
            f"device     {device}",
        ]
        + [f"layer      {m}" for m in student.policy if hasattr(m, "in_features")],
    )

    section(
        "logging",
        [f"tensorboard  logs/distill/{run_name}"],
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
