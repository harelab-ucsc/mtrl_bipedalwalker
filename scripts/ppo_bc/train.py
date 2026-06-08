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
import numpy as np
from ppo_bc_sb3.common.dagger_dataset import DaggerDataset
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure

import time
import os
import subprocess
import threading

# NOTE: pygame + pynput are imported lazily inside play_sound(), NOT at module
# scope. `from pynput import keyboard` raises at import time on a headless Linux
# box (no $DISPLAY), which would make `import train` crash for any non-interactive
# caller (e.g. the bc_ablation sweep). They're only needed for the optional
# end-of-run sound, so defer them to the one place that uses them.

from utils.paths import DATASET_DIR, MODELS_DIR, LOGS_DIR, ROOT
from utils.logging import StandardTBCallback, RewardTermLogger, fmt_duration

# new env that drives task + cmd sampling internally (walk / flamingo / tilt mix).
from wrappers.ppo_bc.ppo_bc_env import RlFTEnv
from mdp.bipedal_walker.tasks import GAIT

from ppo_bc_sb3 import PPO_BC, PpoBcPolicy, load_expert

# all hyperparameters live in train_config.py; pick one with --preset.
from train_config import PRESETS, TrainConfig

# exist_ok=True is deliberate: when several routine processes import this module
# concurrently (the bc_ablation sweep), the old "if not exists: makedirs" pattern
# races and one importer can crash with FileExistsError.
os.makedirs(MODELS_DIR / "ppo_bc", exist_ok=True)
os.makedirs(LOGS_DIR / "ppo_bc", exist_ok=True)
os.makedirs(DATASET_DIR / "ppo_bc", exist_ok=True)

# =========================================


def build_experts(cfg: TrainConfig):
    """Load expert checkpoints once and return a dict[task_name, callable(obs)->act].

    Keys are the directional task names ``resolve_task`` returns for the active
    scheme (gait: walk fwd/bwd, hop fwd/bwd, tilt; onehot: walk fwd/bwd, flamingo,
    tilt). Each callable receives the full RlFTEnv obs (shape [N, n_proprio+2+
    task_bits]) and returns expert actions [N, 4]; it owns its own obs slicing so
    PPO_BC stays agnostic to the env layout. The algorithm routes by direction, so
    each directional callable only ever sees envs of its own direction.
    """
    n_proprio = cfg.n_proprio
    # obs layout: [proprio(n_proprio) | cmd_vel, cmd_tilt | task_bits]

    def _vel_call(expert):
        # walk / hop experts expect [proprio | cmd_vel]; the velocity sign already
        # matches the expert's direction by the time we're routed here.
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
        # directional hops, each velocity-conditioned (hop @ 0 = "flamingo").
        hop_fwd = load_expert(MODELS_DIR / cfg.expert_paths["hop_forward"])
        hop_bwd = load_expert(MODELS_DIR / cfg.expert_paths["hop_backward"])
        return {
            "walk_forward": _vel_call(walk_fwd),
            "walk_backward": _vel_call(walk_bwd),
            "hop_forward": _vel_call(hop_fwd),
            "hop_backward": _vel_call(hop_bwd),
            "tilt": _tilt_call(tilt),
        }

    # onehot legacy: a single "flamingo" task = hop_forward polled at cmd_vel=0.
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


def make_env(cfg: TrainConfig):
    # RlFTEnv subclasses ProprioObsWrapper internally, so we don't need to wrap
    # the raw bipedal walker env in ProprioObsWrapper ourselves.
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


def make_adversarial_eval_env(cfg: TrainConfig) -> RlFTEnv:
    # single, non-vectorized RlFTEnv used only by PPO_BC's internal adversarial
    # eval routine. Mirrors make_env's config (sans Monitor — we only care
    # about per-task time-alive, not SB3 episode bookkeeping).
    return RlFTEnv(
        gym.make("BipedalWalker-v3"),
        ep_time=cfg.ep_time,
        cmd_switching_time=cfg.cmd_switching_time,
        task_switching_time=cfg.task_switching_time,
        # mirror the training env's task config: adversarial selection builds the
        # PMF positionally over allowed_task_mixing, so the eval env must enumerate
        # exactly the training tasks in the same order (single, combination, or any
        # mix). task_switch_replacement is mirrored so the init preflight matches.
        task_switch_replacement=cfg.task_switch_replacement,
        cmd_interp_speed=cfg.cmd_interp_speed,
        cmd_sample_range=cfg.cmd_sample_range,
        cmd_sample_zero=cfg.cmd_sample_zero,
        allowed_task_mixing=list(cfg.allowed_task_mixing),
        use_rew_for_individual_tasks=True,
        hull_x_range=cfg.hull_x_range,
        task_scheme=cfg.task_scheme,
    )


def main(cfg: TrainConfig, extra_callbacks=None, notify=True):
    """Train one PPO_BC run.

    extra_callbacks: optional list of SB3 BaseCallbacks appended to the standard
        set — used by the bc_ablation launcher to inject a progress writer.
    notify: when False, skip the end-of-run macOS notification + sound (the sound
        path spins up a pygame mixer + a pynput keyboard listener, which is a
        footgun under many parallel headless routines).
    """
    print("Loading environments...")

    # bind cfg into the env factory so SubprocVecEnv workers (spawned, not forked)
    # reconstruct the same config without relying on module globals.
    env_fn = partial(make_env, cfg)
    train_env = SubprocVecEnv([env_fn for _ in range(cfg.n_train_envs)])
    eval_env = SubprocVecEnv([env_fn for _ in range(cfg.n_eval_envs)])

    print("Loading experts...")
    experts = build_experts(cfg)

    policy_kwargs = dict(
        hidden_dims=list(cfg.hidden_dims),
        critic_hidden_dims=list(cfg.critic_hidden_dims),
        activation_fn=cfg.activation_fn,
    )

    model = PPO_BC(
        PpoBcPolicy,
        train_env,
        experts=experts,
        task_bits=cfg.task_bits,
        task_scheme=cfg.task_scheme,
        act_var_floor=cfg.act_var_floor,
        bc_coef=cfg.bc_coef,
        bc_batch_size=cfg.bc_batch_size,
        bc_loss_type=cfg.bc_loss_type,
        collect_data=cfg.collect_data,
        adversarial_ag=cfg.adversarial_ag,
        adversarial_eval_env=make_adversarial_eval_env(cfg),
        adversarial_eval_steps_per_task=cfg.adversarial_eval_steps_per_task,
        adversarial_k=cfg.adversarial_k,
        dagger_max_size=cfg.dagger_max_size,
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
        target_kl=cfg.target_kl,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=policy_kwargs,
        device=cfg.device,
    )
    model.set_dataset_save_path(str(DATASET_DIR / cfg.experiment_name))
    model.set_logger(configure(str(LOGS_DIR / cfg.experiment_name), ["tensorboard"]))
    train_env.reset()

    # warm-start actor+critic from a prior PPO_BC zip if requested.
    # set_parameters loads only the policy/optimizer state from the zip —
    # experts/dagger fields stay as wired in this run.
    if cfg.load_model:
        # cfg.load_model is a bare path; MODELS_DIR is added here, not in the config.
        model.set_parameters(
            str(MODELS_DIR / cfg.load_model), exact_match=True, device=cfg.device
        )
        if cfg.init_log_std is not None:
            with torch.no_grad():
                model.policy.log_std.fill_(cfg.init_log_std)

    # load in dagger dataset if specified (npz from a prior dump_dataset call).
    # cfg.load_dataset is a bare path; DATASET_DIR is added here, not in the config.
    if cfg.load_dataset:
        # pass cfg.device explicitly: DaggerDataset.load defaults to "auto", which
        # resolves to cuda on a GPU box and would mismatch the (cpu) policy in the
        # BC loss. The fresh-collected dataset already uses self.device this way.
        model.demo_dataset = DaggerDataset.load(
            str(DATASET_DIR / cfg.load_dataset),
            device=cfg.device, max_size=cfg.dagger_max_size,
        )

    # define callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{str(MODELS_DIR)}/{cfg.experiment_name}/best",
        eval_freq=max(50000 // train_env.num_envs, 1),
        n_eval_episodes=5,
        verbose=0,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(100000 // train_env.num_envs, 1),
        save_path=f"{MODELS_DIR}/{cfg.experiment_name}/",
    )

    # print out model and environment settings
    print_run_info(cfg, train_env, model)

    callbacks = [StandardTBCallback(), RewardTermLogger(), eval_cb, ckpt_cb]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    start_time = time.time()
    model.learn(
        total_timesteps=cfg.timesteps,
        reset_num_timesteps=False,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    # save the final model (best_model is saved by EvalCallback)
    model.save(f"{MODELS_DIR}/{cfg.experiment_name}/final")

    # Explicitly tear down the SubprocVecEnv workers (14 train + 5 eval). Without
    # this they linger until GC, which wastes cores when a caller runs several
    # stages back-to-back in one process (the bc_ablation routine does exactly this).
    train_env.close()
    eval_env.close()

    duration = fmt_duration(time.time() - start_time)
    print(f"Done! Total time: {duration}")
    print(f"Experiment name: {cfg.experiment_name}")

    if notify:
        try:
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'display notification "Finished in {duration}" with title "Training complete" subtitle "{cfg.experiment_name}"',
                ],
                check=False,
            )
        except FileNotFoundError:
            pass  # not on macOS
        try:
            play_sound(ROOT / "assets" / "train_finish.mp3")
        except Exception as e:
            print(f"(skipping play_sound: {e})")


def print_run_info(cfg: TrainConfig, env, model):
    """Echo the full resolved config (+ the actually-built model/env) so a run's
    settings can be eyeballed before training — e.g. confirm bc_loss_type."""
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
    print(f"  experiment  {cfg.experiment_name}")
    print(f"  timesteps   {cfg.timesteps:,}")
    print(f"{'=' * 44}")

    cmd_vel_sw, cmd_tilt_sw = cfg.cmd_switching_time
    section(
        "environment",
        [
            f"train / eval envs   {cfg.n_train_envs} / {cfg.n_eval_envs}",
            f"env                 {env_id}",
            f"obs                 {obs.shape}  {obs.dtype}",
            f"act                 {act.shape}  [{act.low[0]:.1f}, {act.high[0]:.1f}]",
            f"ep_time             {cfg.ep_time}s",
            f"cmd_switch vel/tilt {switch_desc(cmd_vel_sw)} / {switch_desc(cmd_tilt_sw)}",
            f"task_switch         {switch_desc(cfg.task_switching_time)}",
            f"task_switch_replace {cfg.task_switch_replacement}",
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

    adv_lines = [f"adversarial_ag      {cfg.adversarial_ag}"]
    if cfg.adversarial_ag:
        adv_lines += [
            f"eval_steps_per_task {cfg.adversarial_eval_steps_per_task:,}",
            f"adversarial_k       {cfg.adversarial_k}",
        ]
    section("adversarial", adv_lines)

    section(
        "dagger / bc",
        [
            f"bc_coef             {cfg.bc_coef}",
            f"bc_loss_type        {cfg.bc_loss_type}",
            f"bc_batch_size       {cfg.bc_batch_size}",
            f"act_var_floor       {cfg.act_var_floor}",
            f"task_bits           {cfg.task_bits}",
            f"collect_data        {cfg.collect_data}",
            f"dagger_max_size     {cfg.dagger_max_size}",
        ]
        + [f"expert {k}  ->  callable" for k in sorted(model.experts.keys())],
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
            f"target_kl           {cfg.target_kl}",
            f"max_grad_norm       {cfg.max_grad_norm}",
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
            f"actor   in -> {actor} -> {act_out}",
            f"critic  in -> {critic} -> 1",
            f"activation          {cfg.activation_fn.__name__}",
            f"action std (init)   {std}",
        ],
    )

    section(
        "warm-start",
        [
            f"load_model          {cfg.load_model}",
            f"load_dataset        {cfg.load_dataset}",
            f"init_log_std        {cfg.init_log_std}",
        ],
    )

    print(f"\n{'=' * 44}\n")


def play_sound(path):
    # imported here (not at module scope) so headless callers can `import train`
    # without a display — see the note next to the top-of-file imports.
    import pygame
    from pynput import keyboard as kb

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
    parser = argparse.ArgumentParser(description="Train PPO_BC from a named preset.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="pretrain",
        help="which train_config preset to run (default: pretrain)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PRESETS[args.preset]
    print(f"Using preset: {args.preset}  ->  experiment {cfg.experiment_name}")
    main(cfg)
