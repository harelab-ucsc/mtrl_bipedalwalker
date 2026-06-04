# https://arxiv.org/pdf/2505.11164

import os
import argparse
import warnings
import time
import subprocess
import threading

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
    StudentModel,
    OBS_SIZE_V2,
    ACT_SIZE,
)
from mdp.bipedal_walker.tasks import GAIT, resolve_single_task
from train_config import PRESETS, DistillConfig


# onehot task ids: 0 walk forward, 1 walk backward, 2 flamingo, 3 tilt
# gait task ids: index into cfg.gait_tasks (walk fwd/bwd, hop fwd/bwd, tilt)
def configure_env(e: DistillEnv, task_id: int, cfg: DistillConfig):
    """Configure the env's hull reset + command sampling for a given task.

    All sampling knobs (switch/interp speeds, sample ranges, zero-probabilities,
    hull spawn x_range) come from ``cfg`` — see DistillConfig.
    ``cfg.mix_irrelevant_input`` controls whether irrelevant commands are mixed
    in (so the student learns to ignore them) or reset to 0 for clean inputs.
    """
    vel_switch, tilt_switch = cfg.cmd_switching_time
    vel_interp, tilt_interp = cfg.cmd_interp_speed
    vel_zero, tilt_zero = cfg.cmd_sample_zero

    # gait: the GaitTask's own command ranges fully define what's sampled (vel and
    # tilt ranges already zero whatever is irrelevant), so no mix/mask handling.
    if cfg.task_scheme == GAIT:
        t = cfg.gait_tasks[task_id]
        e.config_hull_reset(x_range=cfg.hull_x_range)
        e.config_cmd_vel(
            sample_range=t.cmd_vel_range, interp_speed=vel_interp,
            switch_time=vel_switch, zero_prob=vel_zero,
        )
        e.config_cmd_tilt(
            sample_range=t.cmd_tilt_range, interp_speed=tilt_interp,
            switch_time=tilt_switch, zero_prob=tilt_zero,
        )
        return

    mix = cfg.mix_irrelevant_input
    (vel_lo, vel_hi), tilt_range = cfg.cmd_sample_range

    def cmd_vel(sample_range):
        e.config_cmd_vel(sample_range=sample_range, interp_speed=vel_interp, switch_time=vel_switch, zero_prob=vel_zero)

    def cmd_tilt():
        e.config_cmd_tilt(sample_range=tilt_range, interp_speed=tilt_interp, switch_time=tilt_switch, zero_prob=tilt_zero)

    e.config_hull_reset(x_range=cfg.hull_x_range)

    if task_id == 0:  # walk forward
        cmd_vel((0.0, vel_hi))
        cmd_tilt() if mix else e.config_cmd_tilt(zero_prob=1)  # mix random tilt / reset to 0
    elif task_id == 1:  # walk backward
        cmd_vel((vel_lo, 0.0))
        cmd_tilt() if mix else e.config_cmd_tilt(zero_prob=1)
    elif task_id == 2:  # flamingo
        if mix:  # mix in random tilt and velocity commands
            cmd_vel((vel_lo, vel_hi))
            cmd_tilt()
        else:  # reset tilt and velocity command to 0 for clean input
            e.config_cmd_vel(zero_prob=1)
            e.config_cmd_tilt(zero_prob=1)
    elif task_id == 3:  # tilt
        cmd_tilt()
        cmd_vel((vel_lo, vel_hi)) if mix else e.config_cmd_vel(zero_prob=1)  # mix random vel / reset to 0


def main(cfg: DistillConfig):
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type  # type: ignore
    else:
        device = "cpu"

    experiment_name = cfg.experiment_name

    print("Loading environments...")
    env = DistillEnv(make("BipedalWalker-v3", render_mode=None), ep_time=cfg.ep_time)
    eval_env = DistillEnv(make("BipedalWalker-v3", render_mode=None), ep_time=cfg.eval_ep_time)

    gait = cfg.task_scheme == GAIT

    # load models. Gait keys experts by directional name (resolve_task returns it);
    # onehot keeps the legacy index-ordered list.
    print("Loading experts...")
    if gait:
        gait_tasks = list(cfg.gait_tasks)
        task_names = [t.name for t in gait_tasks]
        EXPERTS = {
            name: PPO.load(MODELS_DIR / p, env=None, device="cpu")
            for name, p in cfg.gait_expert_paths.items()
        }
    else:
        task_names = list(cfg.task_names)
        EXPERT_MODELS = [
            PPO.load(MODELS_DIR / p, env=None, device="cpu") for p in cfg.expert_paths
        ]
    print("Loading student...")
    student = cfg.make_student()

    n_tasks = len(task_names)

    def task_bits(task_id: int):
        """The 3 obs bits for a task id (gait_bits under gait, one-hot under onehot)."""
        if gait:
            return gait_tasks[task_id].gait_bits
        if task_id in (0, 1):
            return [1, 0, 0]  # walk
        if task_id == 2:
            return [0, 1, 0]  # flamingo
        return [0, 0, 1]  # tilt

    def forwardExpert(obs: np.ndarray, task_id: int, cmd_vel: float, cmd_tilt: float = 0.0) -> np.ndarray:
        if gait:
            # route by the directional task resolve_task picks from the runtime
            # commands (handles walk-vs-tilt and the 0-vel → forward-expert rule).
            spec = resolve_single_task(gait_tasks[task_id].gait_bits, cmd_vel, cmd_tilt, GAIT)
            assert spec is not None, "single gait tasks always resolve to one expert"
            cmd = cmd_tilt if spec.name == "tilt" else cmd_vel
            action, _ = EXPERTS[spec.name].predict(np.append(obs, cmd), deterministic=True)
            return action

        # --- onehot (legacy) ---
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
        if cfg.adversarial_task_select:
            assert 0 <= k <= 1
            w = [max(t) - i for i in t]
            sum_w = sum(w)
            U = [1 / len(t)] * len(t)
            P = [U[i] if sum_w == 0 else w[i] / sum_w for i in range(len(t))]
            return [k * p_i + (1 - k) * u_i for p_i, u_i in zip(P, U)]
        else:
            return [1 / len(t)] * len(t)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.decay)
    scheduler = None
    if cfg.use_scheduler:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.sched_restart_iters * cfg.epoch, eta_min=5e-4
        )

    D: list[tuple[np.ndarray, np.ndarray, int]] = []

    writer = SummaryWriter(log_dir=str(LOGS_DIR / experiment_name))
    print_run_info(
        student,
        OBS_SIZE_V2,
        ACT_SIZE,
        device,
        cfg.N,
        cfg.T,
        cfg.epoch,
        cfg.batch_size,
        cfg.lr,
        cfg.decay,
        experiment_name,
    )

    # eval routine, no noise
    def evaluate(step: int):
        student.eval()
        student.to("cpu")
        task_losses = []
        all_time_alive = []
        with torch.no_grad():
            for task_id, task_name in enumerate(task_names):
                configure_env(eval_env, task_id, cfg)
                obs, info = eval_env.reset()
                cmd_vel = info["cmd"]["x_vel"]
                cmd_tilt = info["cmd"]["tilt"]
                done = False
                step_losses = []
                time_alive = []
                alive = 0

                for _ in range(cfg.t_eval):
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
                    obs_s = StudentModel.obs(obs, cmd_vel, cmd_tilt, task_bits(task_id))
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

    # create all the necessary folders (builds the whole rudin[_adv]/distill/x.y.z tree)
    os.makedirs(MODELS_DIR / experiment_name, exist_ok=True)
    os.makedirs(LOGS_DIR / experiment_name, exist_ok=True)

    # DAgger
    start_time = time.time()
    bar = tqdm(total=(cfg.N * cfg.T) + (cfg.N * cfg.epoch), desc="Training", ascii=" ░▒█")
    task_live_time = [1.0] * n_tasks

    for n in range(cfg.N):
        iter_start = time.time()
        Di: list[tuple[np.ndarray, np.ndarray, int]] = []

        # 1. collect trajectories under current student policy (noisy rollout,
        #    clean expert labels)
        student.to("cpu")
        student.eval()
        time_alive = []
        with torch.no_grad():
            done = True
            alive = 0

            for _ in range(cfg.T):
                if done:
                    current_task = int(
                        np.random.choice(n_tasks, p=getTaskPMF(task_live_time, cfg.adversarial_k))
                    )
                    configure_env(env, current_task, cfg)
                    obs, info = env.reset()
                    cmd_vel = info["cmd"]["x_vel"]
                    cmd_tilt = info["cmd"]["tilt"]
                    alive = 0

                # CLEAN expert action — saved as the training label
                act_expert = forwardExpert(obs, current_task, cmd_vel, cmd_tilt)

                obs_s = StudentModel.obs(obs, cmd_vel, cmd_tilt, task_bits(current_task))
                act_student = student(torch.tensor(obs_s, dtype=torch.float32)).numpy()

                # additive diagonal gaussian noise on the EXECUTED action only,
                # for state coverage / exploration (NOT on the saved label, NOT in eval)
                act_exec = act_student + np.random.normal(0.0, cfg.act_var ** 0.5, act_student.shape)

                alive += 1
                obs, _, term, trunc, info = env.step(act_exec)
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
        if cfg.n_active is not None and len(D) > cfg.n_active * cfg.T:
            # linearly increasing weights: oldest sample is around 0, newest = highest
            w = np.arange(1, len(D) + 1, dtype=np.float64)
            w /= w.sum()
            idx = np.random.choice(len(D), size=cfg.n_active * cfg.T, replace=False, p=w)
            obs_arr = np.array(obs_list)[idx]
            act_arr = np.array(act_list)[idx]
            task_ids_all = tuple(np.array(task_ids_all)[idx])
        else:
            obs_arr = np.array(obs_list)
            act_arr = np.array(act_list)

        x_full = torch.tensor(obs_arr, dtype=torch.float32)
        y_full = torch.tensor(act_arr, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(x_full, y_full), batch_size=cfg.batch_size, shuffle=True
        )

        student.to(device)
        student.train()

        for epoch in range(cfg.epoch):
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

            writer.add_scalar("train/loss", epoch_loss / len(loader), n * cfg.epoch + epoch)
            if scheduler is not None:
                scheduler.step()
            bar.update(1)

        # log per-iteration scalars
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], n * cfg.epoch)
        writer.add_scalar("train/dataset_size", len(D), n * cfg.epoch)
        if time_alive:
            writer.add_scalar(
                "train/avg_time_alive", float(np.mean(time_alive)), n * cfg.epoch
            )

        # per-expert training loss on full accumulated dataset
        student.eval()
        student.to("cpu")
        with torch.no_grad():
            pred_all = student(x_full)
            total_D = len(task_ids_all)
            for task_id, task_name in enumerate(task_names):
                indices = [i for i, t in enumerate(task_ids_all) if t == task_id]
                if indices:
                    idx_t = torch.tensor(indices)
                    writer.add_scalar(
                        f"train/loss_{task_name}",
                        F.mse_loss(pred_all[idx_t], y_full[idx_t]).item(),
                        n * cfg.epoch,
                    )
                writer.add_scalar(
                    f"train/task_pct_{task_name}",
                    len(indices) / total_D,
                    n * cfg.epoch,
                )

        # 4. eval
        _, _, task_live_time = evaluate(n * cfg.epoch)

        # 5. save periodic checkpoints
        if n % cfg.ckpt_int == 0 or n == cfg.N - 1:
            torch.save(
                {
                    "policy": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                str(MODELS_DIR / experiment_name / f"distill_{n}.pt"),
            )

        iter_time = time.time() - iter_start
        iter_per_s = 1.0 / iter_time
        writer.add_scalar("train/iter_per_s", iter_per_s, n)
        writer.add_scalar("train/iter_time_s", iter_time, n)
        bar.set_postfix(iter_s=f"{iter_per_s:.3f}")

    bar.close()
    writer.close()

    # save the final model
    torch.save(
        {
            "policy": student.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        str(MODELS_DIR / experiment_name / "final.pt"),
    )

    duration = fmt_duration(time.time() - start_time)
    print(f"\nDone! Total time: {duration}")

    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "Finished in {duration}" with title "Distillation complete" subtitle "{experiment_name}"',
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DAgger distillation student from a named preset.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="adversarial",
        help="which train_config preset to run (default: adversarial)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="__array_wrap__ must accept context and return_scalar arguments",
        category=DeprecationWarning,
    )
    args = parse_args()
    cfg = PRESETS[args.preset]
    print(f'Using preset "{args.preset}"  ->  experiment "{cfg.experiment_name}"')
    main(cfg)
