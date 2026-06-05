"""
scripts/rlft/finetune.py
========================

RL fine-tuning for the Rudin baseline (pure RL, no behavior cloning).

Warm-starts from a pretrained-critic zip (actor + critic, produced by
scripts/rlft/pretrain_critic.py) and continues training in the same pure-RL
RlFTEnv, but with the actor unfrozen and **stricter objective clipping + a
smaller, decaying LR** so RL refines the policy without diverging from the
distilled start. The network architecture is inherited from the loaded zip. The
env + PPO settings mirror ppo_bc's 2.x.x RL presets for a fair baseline.

The adversarial (``rudin_adv``) lineage additionally runs adversarial task
selection during RL via ``AdversarialTaskCallback`` (difficulty-weighted task
sampling, a stock-PPO port of PPO_BC's eval_expert_task_performance), enabled by
``cfg.adversarial_ag``. The plain lineage samples tasks uniformly.

Saves to ``models/rudin[_adv]/{combination,switching,comb_switching}/<version>/``:
  - ``best/best_model.zip``     (best by eval reward; what play/compare load)
  - ``rl_model_*_steps.zip``    (periodic checkpoints; what eval/time_alive loads)
  - ``final.zip``               (last checkpoint)

Run:  python scripts/rlft/finetune.py --preset switching_200a
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
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import obs_as_tensor

import time
import subprocess

from utils.paths import MODELS_DIR, LOGS_DIR
from utils.logging import StandardTBCallback, RewardTermLogger, fmt_duration
from wrappers.ppo_bc.ppo_bc_env import RlFTEnv

from finetune_config import PRESETS, FinetuneConfig

# =========================================


def make_env(cfg: FinetuneConfig):
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


def make_adversarial_eval_env(cfg: FinetuneConfig) -> RlFTEnv:
    # Single, non-vectorized, un-Monitored RlFTEnv used only by the adversarial
    # task-selection callback for per-task time-alive scoring. Mirrors make_env's
    # task config exactly: the PMF is built positionally over allowed_task_mixing,
    # so the eval env must enumerate the same tasks in the same order. (Port of
    # scripts/ppo_bc/train.py:make_adversarial_eval_env, adapted to FinetuneConfig.)
    return RlFTEnv(
        gym.make("BipedalWalker-v3"),
        ep_time=cfg.ep_time,
        cmd_switching_time=cfg.cmd_switching_time,
        task_switching_time=cfg.task_switching_time,
        task_switch_replacement=cfg.task_switch_replacement,
        cmd_interp_speed=cfg.cmd_interp_speed,
        cmd_sample_range=cfg.cmd_sample_range,
        cmd_sample_zero=cfg.cmd_sample_zero,
        allowed_task_mixing=list(cfg.allowed_task_mixing),
        use_rew_for_individual_tasks=True,  # always on for eval
        hull_x_range=cfg.hull_x_range,
        task_scheme=cfg.task_scheme,
    )


def _task_pmf_from_scores(scores: list[float], k: float) -> list[float]:
    """Weight by (max - score) so the worst (hardest, lowest time-alive) task gets
    the most mass, then mix with uniform by ``k``. Kept byte-identical to
    src/ppo_bc_sb3/common/on_policy_algorithm.py:_task_pmf_from_scores for fairness
    with the ppo_bc method."""
    w = [max(scores) - s for s in scores]
    sum_w = sum(w)
    U = [1.0 / len(scores)] * len(scores)
    P = [U[i] if sum_w == 0 else w[i] / sum_w for i in range(len(scores))]
    return [k * p + (1.0 - k) * u for p, u in zip(P, U)]


class AdversarialTaskCallback(BaseCallback):
    """Difficulty-weighted task sampling for RL fine-tuning — a stock-PPO port of
    PPO_BC's ``OnPolicyAlgorithm.eval_expert_task_performance``. Once per rollout,
    score the current policy on each allowed task by mean time-alive on a single
    isolated eval env, convert to a PMF (hardest task gets the most mass, blended
    with uniform by ``k``), and broadcast it to every training env via
    ``set_task_sample_probs``. Expert-independent: only needs the policy + eval env.

    Scoring runs on ``_on_rollout_end`` (the natural SB3 hook), so it scores the
    pre-update policy of the iteration and the PMF takes effect on the next rollout
    — a one-iteration lag vs PPO_BC's post-update timing, negligible for difficulty
    estimation."""

    def __init__(self, eval_env: RlFTEnv, steps_per_task: int, k: float, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.steps_per_task = steps_per_task
        self.k = k

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        env = self.eval_env
        tasks = list(env._allowed_tasks)
        low, high = env.action_space.low, env.action_space.high

        self.model.policy.set_training_mode(False)
        scores: list[float] = []
        for task in tasks:
            env.set_forced_task(task)
            obs, _ = env.reset()
            times: list[int] = []
            alive = 0
            for _ in range(self.steps_per_task):
                with torch.no_grad():
                    act_t, _, _ = self.model.policy(
                        obs_as_tensor(obs[None], self.model.device)
                    )
                act = np.clip(act_t.cpu().numpy()[0], low, high)
                obs, _, term, trunc, _ = env.step(act)
                if term or trunc:
                    times.append(alive)
                    alive = 0
                    obs, _ = env.reset()
                else:
                    alive += 1
            times.append(alive)
            scores.append(float(np.mean(times)))
        env.set_forced_task(None)
        self.model.policy.set_training_mode(True)

        probs = _task_pmf_from_scores(scores, self.k)
        # env_method routes through gym.Wrapper.__getattr__ to the inner RlFTEnv;
        # set_attr would only touch the outer Monitor wrapper.
        self.training_env.env_method("set_task_sample_probs", tuple(probs))
        for task, s, p in zip(tasks, scores, probs):
            self.logger.record(f"adversarial/time_alive_{task.name}", s)
            self.logger.record(f"adversarial/prob_{task.name}", p)


def main(cfg: FinetuneConfig):
    assert cfg.load_pretrained_from, "cfg.load_pretrained_from is not set — point it at a pretrained-critic zip."
    pretrained = MODELS_DIR / cfg.load_pretrained_from
    assert pretrained.exists(), f"pretrained critic zip not found: {pretrained}"

    experiment_name = cfg.experiment_name

    print("Loading environments...")
    env_fn = partial(make_env, cfg)
    train_env = SubprocVecEnv([env_fn for _ in range(cfg.n_train_envs)])
    eval_env = SubprocVecEnv([env_fn for _ in range(cfg.n_eval_envs)])

    print(f"Loading pretrained critic from {pretrained}...")
    # custom_objects overrides the pretrain-time PPO hyperparams with the
    # fine-tune ones (tighter clip, lower LR, etc.). The policy/critic weights
    # come from the zip; the actor is trainable again after a fresh policy build.
    model = PPO.load(
        pretrained,
        env=train_env,
        device=cfg.device,
        custom_objects={
            "learning_rate": cfg.learning_rate,
            "clip_range": cfg.clip_range,
            "n_epochs": cfg.n_epochs,
            "n_steps": cfg.n_steps,
            "batch_size": cfg.batch_size,
            "ent_coef": cfg.ent_coef,
            "vf_coef": cfg.vf_coef,
            "gamma": cfg.gamma,
            "gae_lambda": cfg.gae_lambda,
            "max_grad_norm": cfg.max_grad_norm,
        },
    )

    if cfg.init_log_std is not None:
        with torch.no_grad():
            model.policy.log_std.fill_(cfg.init_log_std)

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
    # adv lineage: difficulty-weighted task sampling during RL. The standard
    # EvalCallback above stays on the uniform eval env (best-model selection
    # unbiased); this drives the *training* env task distribution only.
    adv_eval_env = None
    if cfg.adversarial_ag:
        adv_eval_env = make_adversarial_eval_env(cfg)
        callbacks.append(
            AdversarialTaskCallback(
                adv_eval_env,
                cfg.adversarial_eval_steps_per_task,
                cfg.adversarial_k,
            )
        )

    print_run_info(cfg, train_env, model, experiment_name, pretrained)

    print(f"Starting fine-tuning ({cfg.timesteps:,} timesteps)...")
    start = time.time()
    model.learn(
        total_timesteps=cfg.timesteps,
        reset_num_timesteps=True,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    model.save(f"{MODELS_DIR}/{experiment_name}/final")

    duration = fmt_duration(time.time() - start)
    print(f"Done! Total time: {duration}")
    print(f"Experiment name: {experiment_name}")

    try:
        subprocess.run(
            [
                "osascript", "-e",
                f'display notification "Finished in {duration}" with title "RLFT fine-tune complete" subtitle "{experiment_name}"',
            ],
            check=False,
        )
    except FileNotFoundError:
        pass  # not on macOS

    train_env.close()
    eval_env.close()
    if adv_eval_env is not None:
        adv_eval_env.close()


def print_run_info(cfg, env, model, experiment_name, pretrained):
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
    print(f"  finetune         {experiment_name}")
    print(f"  pretrained from  {pretrained}")
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
            f"adversarial_ag      {cfg.adversarial_ag}"
            + (
                f"  (k={cfg.adversarial_k}, {cfg.adversarial_eval_steps_per_task} steps/task)"
                if cfg.adversarial_ag
                else ""
            ),
        ],
    )

    section(
        "ppo (fine-tune)",
        [
            f"device            {model.device}",
            f"lr                {lr_desc(cfg.learning_rate)}",
            f"clip_range        {cfg.clip_range}",
            f"n_steps           {model.n_steps}",
            f"batch_size        {model.batch_size}",
            f"n_epochs          {model.n_epochs}",
            f"gamma / lambda    {model.gamma} / {model.gae_lambda}",
            f"vf_coef / ent     {model.vf_coef} / {model.ent_coef}",
            f"init_log_std      {cfg.init_log_std}",
        ],
    )

    def net_summary(net):
        return " -> ".join(str(l.out_features) for l in net if hasattr(l, "out_features"))

    actor = net_summary(p.mlp_extractor.policy_net)
    critic = net_summary(p.mlp_extractor.value_net)
    section(
        "network (from zip)",
        [
            f"actor   in -> {actor} -> {p.action_net.out_features}",
            f"critic  in -> {critic} -> 1",
        ],
    )

    print(f"\n{'=' * 44}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL fine-tune a Rudin-baseline critic-pretrained model.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="switching_200a",
        help="which finetune_config preset to run (default: switching_200a)",
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
