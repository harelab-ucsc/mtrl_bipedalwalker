# https://arxiv.org/pdf/2505.11164

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

import pygame
from pynput import keyboard as kb
from tqdm import tqdm
from utils.paths import MODELS_DIR, LOGS_DIR, ROOT
from utils.logging import fmt_duration
from wrappers.bipedal_walker.distill_env import DistillEnv

EXPERT_MODEL_PATHS = [
    "experts/walk_forward",
    "experts/walk_backward",
    "experts/hop_forward",
    "experts/hop_backward",
]

TASK_NAMES = ["walk_forward", "walk_backward", "hop_forward", "hop_backward"]

RUN_NAME = "distill" + datetime.today().strftime("-%H_%M_%S-%Y_%m_%d")

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

    _tasks = {1: "walk forward", 2: "walk backward", 3: "hop forward", 4: "hop backward"}
    env = DistillEnv(make("BipedalWalker-v3", render_mode=None), ep_time=10, tasks=_tasks)
    eval_env = DistillEnv(make("BipedalWalker-v3", render_mode=None), ep_time=10, tasks=_tasks)

    # load expert models
    print("Loading experts...")
    EXPERT_MODELS = [
        PPO.load(MODELS_DIR / i, env=None, device="cpu") for i in EXPERT_MODEL_PATHS
    ]

    BASE_OBS_SIZE = 14
    OBS_SIZE = BASE_OBS_SIZE + 3  # cmd_vel + 2 one-hot task bits
    ACT_SIZE = 4

    # student model
    class StudentModel(nn.Module):
        def __init__(self, obs_size: int, act_size: int):
            super().__init__()
            self.policy = nn.Sequential(
                nn.Linear(obs_size, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, act_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.policy(x)

    student = StudentModel(OBS_SIZE, ACT_SIZE)

    # ---- helpers ----

    def configureEnv(e: DistillEnv, task_id: int):
        if task_id == 0 or task_id == 2:  # walk / hop forward
            x_range = (0.0, 40.0)
            vel_range = (0.0, 5.0)
        else:                              # walk / hop backward
            x_range = (40.0, 80.0)
            vel_range = (-5.0, 0.0)
        e.set_task(task_id + 1)  # tasks dict is 1-indexed
        e.config_hull_reset(x_range=x_range)
        e.config_cmd_vel(sample_range=vel_range, interp_time=0.5)

    def forwardExpert(obs: np.ndarray, task_id: int, cmd_vel: float) -> np.ndarray:
        action, _ = EXPERT_MODELS[task_id].predict(np.append(obs, cmd_vel), deterministic=True)
        return action

    def studentObs(obs: np.ndarray, task_id: int, cmd_vel: float) -> np.ndarray:
        task_spec = [1, 0] if task_id < 2 else [0, 1]  # walk=10, hop=01
        return np.concatenate([obs, [cmd_vel], task_spec])

    # ---- DAgger hyperparams ----

    T = 1000        # env steps per DAgger iteration
    N = 20          # DAgger iterations
    EPOCH = 30      # training epochs per iteration
    BATCH_SIZE = 256
    LR = 1e-3
    DECAY = 1e-2
    T_EVAL = 300    # eval steps per expert task

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=DECAY)

    D: list[tuple[np.ndarray, np.ndarray, int]] = []

    writer = SummaryWriter(log_dir=str(LOGS_DIR / "distill" / RUN_NAME))
    print_run_info(student, OBS_SIZE, ACT_SIZE, device, N, T, EPOCH, BATCH_SIZE, LR, DECAY, RUN_NAME)

    # ---- eval routine ----

    def evaluate(n: int):
        student.eval()
        student.to("cpu")
        task_losses = []
        with torch.no_grad():
            for task_id, task_name in enumerate(TASK_NAMES):
                configureEnv(eval_env, task_id)
                obs, info = eval_env.reset()
                cmd_vel = info["cmd"]["x_vel"]
                done = False
                step_losses = []

                for _ in range(T_EVAL):
                    if done:
                        obs, info = eval_env.reset()
                        cmd_vel = info["cmd"]["x_vel"]
                        done = False

                    act_expert = forwardExpert(obs, task_id, cmd_vel)
                    obs_s = studentObs(obs, task_id, cmd_vel)
                    pred = student(torch.tensor(obs_s, dtype=torch.float32))
                    target = torch.tensor(act_expert, dtype=torch.float32)

                    step_losses.append(F.mse_loss(pred, target).item())

                    obs, _, term, trunc, info = eval_env.step(pred.numpy())
                    cmd_vel = info["cmd"]["x_vel"]
                    done = term or trunc

                task_loss = float(np.mean(step_losses))
                task_losses.append(task_loss)
                writer.add_scalar(f"eval/loss_{task_name}", task_loss, n)

        writer.add_scalar("eval/loss_total", float(np.mean(task_losses)), n)

    # ---- main loop ----

    start_time = time.time()
    bar = tqdm(total=(N * T) + (N * EPOCH), desc="Training", ascii=" ░▒█")

    for n in range(N):
        Di: list[tuple[np.ndarray, np.ndarray, int]] = []

        # 1. collect trajectories under current student policy
        student.to("cpu")
        student.eval()
        with torch.no_grad():
            done = True
            for _ in range(T):
                if done:
                    current_task = int(np.random.choice(4))
                    configureEnv(env, current_task)
                    obs, info = env.reset()
                    cmd_vel = info["cmd"]["x_vel"]

                act_expert = forwardExpert(obs, current_task, cmd_vel)
                obs_s = studentObs(obs, current_task, cmd_vel)
                act_student = student(torch.tensor(obs_s, dtype=torch.float32)).numpy()

                obs, _, term, trunc, info = env.step(act_student)
                cmd_vel = info["cmd"]["x_vel"]
                done = term or trunc

                Di.append((obs_s, act_expert, current_task))
                bar.update(1)

        # 2. aggregate dataset
        D += Di

        # 3. train student on full D
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
                epoch_loss += loss.item()

            writer.add_scalar("train/loss", epoch_loss / len(loader), n * EPOCH + epoch)
            bar.update(1)

        # log per-iteration scalars
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], n)
        writer.add_scalar("train/dataset_size", len(D), n)

        # per-expert training loss on full accumulated dataset
        student.eval()
        student.to("cpu")
        with torch.no_grad():
            pred_all = student(x_full)
            for task_id, task_name in enumerate(TASK_NAMES):
                indices = [i for i, t in enumerate(task_ids_all) if t == task_id]
                if indices:
                    idx_t = torch.tensor(indices)
                    writer.add_scalar(
                        f"train/loss_{task_name}",
                        F.mse_loss(pred_all[idx_t], y_full[idx_t]).item(),
                        n,
                    )

        # 4. eval
        evaluate(n)

    bar.close()
    writer.close()

    duration = fmt_duration(time.time() - start_time)
    print(f"\nDone! Total time: {duration}")

    try:
        subprocess.run(
            [
                "osascript", "-e",
                f'display notification "Finished in {duration}" with title "Distillation complete" subtitle "{RUN_NAME}"',
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
