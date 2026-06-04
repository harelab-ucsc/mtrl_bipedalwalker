"""
scripts/rlft/plot_reward_landscape.py
=====================================

Analyze the reward / return landscape of a distilled or finetuned model on the
v2 RlFTEnv. Kept as an occasional analysis tool — not part of the training path.

2x2 figure layout:
  (0,0)  Raw step reward distributions      (per task + combined)
  (1,0)  Standardised reward distributions  (each task -> N(0,1), combined)
  (0,1)  Combined return distribution        from raw rewards
  (1,1)  Combined return distribution        after per-task reward standardisation

Returns are computed over full episode sequences — G_t = r_t + gamma*G_{t+1} —
so task switching within an episode is handled correctly. The "after scaling"
returns standardise each r_t by its task's statistics before computing G_t.

Migrated from the old walk/hop (17-dim) setup to the v2 4-task system: rewards
are bucketed by ``info["task_name"]`` (walk_forward / walk_backward / flamingo /
tilt / combos), so the breakdown follows whatever ``ALLOWED_TASK_MIXING`` samples.

Interactive: a checkbox toggles log Y scale on all four subplots.
"""

from collections import OrderedDict

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from scipy.stats import gaussian_kde
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import tqdm

from utils.paths import MODELS_DIR
from wrappers.ppo_bc.ppo_bc_env import RlFTEnv
from mdp.bipedal_walker.rlft_policy import RlFTPolicy
from mdp.bipedal_walker.student import HIDDEN_BC
from mdp.bipedal_walker.tasks import SINGLE_TASKS

# =========================================
# Configuration — edit here
# =========================================

# "distill" -> build a PPO shell, load raw actor weights from a distilled .pt
# "ppo"     -> load a fully-trained PPO .zip (finetuned or pretrained critic)
MODEL_SOURCE = "distill"

# Used when MODEL_SOURCE == "distill" (bare path under MODELS_DIR to the student .pt).
DISTILL_CKPT = "rudin/distill/1.0.0/best.pt"

# Used when MODEL_SOURCE == "ppo" (models/-relative).
EXPERIMENT_NAME = "rudin/finetuned/1.0.0"
MODEL_CHECKPOINT = "best/best_model"

# network arch — actor must match the distilled student (HIDDEN_BC).
HIDDEN_DIMS = list(HIDDEN_BC)
CRITIC_HIDDEN_DIMS = [1024, 512, 512, 256, 256]

# Which tasks to sample while collecting (drives the per-task breakdown).
ALLOWED_TASK_MIXING = list(SINGLE_TASKS)

# Step reward samples to gather per task (each bucket is capped at this).
STEPS_PER_TASK = 1_000_000

# Parallel envs for collection.
N_COLLECT_ENVS = 14 * 4

# Discount factor for return computation.
GAMMA = 0.9

# RlFTEnv settings.
EP_TIME = 10
TASK_SWITCHING_TIME = 5.0
CMD_SWITCHING_TIME = (2.0, 3.0)

# =========================================


def make_eval_env() -> RlFTEnv:
    base = gym.make("BipedalWalker-v3")
    return RlFTEnv(
        base,
        ep_time=EP_TIME,
        task_switching_time=TASK_SWITCHING_TIME,
        cmd_switching_time=CMD_SWITCHING_TIME,
        allowed_task_mixing=ALLOWED_TASK_MIXING,
        use_rew_for_individual_tasks=True,
    )


def _load_student_actor(model: PPO, student_path) -> None:
    """Copy a distilled StudentModel's trunk + head into the PPO actor."""
    student_sd = torch.load(student_path, map_location="cpu", weights_only=False)["policy"]
    layer_idx = [int(k.split(".")[1]) for k in list(student_sd.keys())[::2]]
    mlp_ext = model.policy.mlp_extractor
    action_net = model.policy.action_net
    with torch.no_grad():
        for idx in layer_idx[:-1]:
            mlp_ext.policy_net[idx].weight.copy_(student_sd[f"policy.{idx}.weight"])  # type: ignore
            mlp_ext.policy_net[idx].bias.copy_(student_sd[f"policy.{idx}.bias"])  # type: ignore
        action_net.weight.copy_(student_sd[f"policy.{layer_idx[-1]}.weight"])
        action_net.bias.copy_(student_sd[f"policy.{layer_idx[-1]}.bias"])


def load_model(env: RlFTEnv) -> PPO:
    if MODEL_SOURCE == "distill":
        model = PPO(
            RlFTPolicy,
            env=env,
            policy_kwargs=dict(
                hidden_dims=HIDDEN_DIMS,
                critic_hidden_dims=CRITIC_HIDDEN_DIMS,
                activation_fn=torch.nn.ELU,
            ),
        )
        student_path = MODELS_DIR / DISTILL_CKPT
        print(f"  Loading distilled weights: {student_path}")
        _load_student_actor(model, student_path)
    else:
        model_path = MODELS_DIR / f"{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}.zip"
        print(f"  Loading PPO model: {model_path}")
        model = PPO.load(model_path, env=env, device="cpu")
    return model


def _discounted_returns(rewards: list[float]) -> list[float]:
    G, rets = 0.0, []
    for r in reversed(rewards):
        G = r + GAMMA * G
        rets.append(G)
    rets.reverse()
    return rets


# A completed episode: list of (reward, task_name) pairs.
Episode = list[tuple[float, str]]


def collect(model: PPO) -> tuple[OrderedDict[str, np.ndarray], list[Episode]]:
    """Roll out across N_COLLECT_ENVS parallel envs, keeping full episode
    sequences. Buckets per-step rewards by task name (capped at STEPS_PER_TASK)
    and returns the per-task reward arrays + all completed episodes."""
    vec_env = SubprocVecEnv([make_eval_env] * N_COLLECT_ENVS)

    buckets: OrderedDict[str, list[float]] = OrderedDict()
    ep_bufs: list[Episode] = [[] for _ in range(N_COLLECT_ENVS)]
    completed: list[Episode] = []

    n_tasks = len(ALLOWED_TASK_MIXING)
    target = STEPS_PER_TASK * n_tasks
    max_steps = target * 5  # safety so a rarely-sampled task can't hang us

    obs = vec_env.reset()
    pbar = tqdm(total=target, unit="steps", desc="Collecting")
    prev = 0
    total = 0

    def done() -> bool:
        if not buckets:
            return False
        return min(len(v) for v in buckets.values()) >= STEPS_PER_TASK and len(buckets) >= n_tasks

    while not done() and total < max_steps:
        actions, _ = model.predict(obs, deterministic=True)  # type: ignore
        obs, rewards, dones, infos = vec_env.step(actions)
        total += N_COLLECT_ENVS

        for i in range(N_COLLECT_ENVS):
            name = infos[i]["task_name"]
            r = float(rewards[i])
            b = buckets.setdefault(name, [])
            if len(b) < STEPS_PER_TASK:
                b.append(r)
            ep_bufs[i].append((r, name))
            if dones[i]:
                completed.append(ep_bufs[i])
                ep_bufs[i] = []

        filled = sum(min(len(v), STEPS_PER_TASK) for v in buckets.values())
        pbar.update(min(filled, target) - prev)
        prev = min(filled, target)

    pbar.close()
    vec_env.close()
    return (
        OrderedDict((k, np.array(v[:STEPS_PER_TASK])) for k, v in buckets.items()),
        completed,
    )


def _task_stats(buckets: OrderedDict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """mean/std per task name (for standardisation)."""
    return {k: (float(v.mean()), float(v.std())) for k, v in buckets.items()}


def compute_returns(
    episodes: list[Episode], stats: dict[str, tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    raw_all: list[float] = []
    scaled_all: list[float] = []
    for ep in tqdm(episodes, desc="Returns"):
        raw_rews = [r for r, _ in ep]
        scaled_rews = []
        for r, name in ep:
            mu, std = stats.get(name, (0.0, 1.0))
            scaled_rews.append((r - mu) / (std + 1e-8))
        raw_all.extend(_discounted_returns(raw_rews))
        scaled_all.extend(_discounted_returns(scaled_rews))
    return np.array(raw_all), np.array(scaled_all)


# ── plotting ──────────────────────────────────────────────────────────────────

_TOTAL_COLOR = "#4CAF50"
_RETURN_COLOR = "#9C27B0"
_TASK_CMAP = plt.get_cmap("tab10")


def _kde_curve(ax, data: np.ndarray, color, label: str):
    if data.size < 2 or np.allclose(data, data[0]):
        return
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 2000)
    y = kde(x)
    ax.plot(x, y, color=color, linewidth=2, label=label)
    ax.fill_between(x, y, alpha=0.12, color=color)


def _draw_reward_raw(ax, buckets: OrderedDict[str, np.ndarray]) -> None:
    total = np.concatenate(list(buckets.values()))
    for i, (name, vals) in enumerate(buckets.items()):
        _kde_curve(ax, vals, _TASK_CMAP(i % 10), f"{name}  mu={vals.mean():.3f}")
    _kde_curve(ax, total, _TOTAL_COLOR, f"Total  mu={total.mean():.3f}  s2={total.var():.3f}")
    ax.set_xlabel("Step reward")
    ax.set_ylabel("Density")
    ax.set_title("Step reward distributions (raw)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def _draw_reward_std(
    ax, buckets: OrderedDict[str, np.ndarray], stats: dict[str, tuple[float, float]]
) -> None:
    scaled = []
    for i, (name, vals) in enumerate(buckets.items()):
        mu, std = stats[name]
        s = (vals - mu) / (std + 1e-8)
        scaled.append(s)
        _kde_curve(ax, s, _TASK_CMAP(i % 10), f"{name} (standardised)")
    combined = np.concatenate(scaled)
    _kde_curve(ax, combined, _TOTAL_COLOR, f"Combined  mu={combined.mean():.3f}  s2={combined.var():.3f}")
    ax.set_xlabel("Standardised step reward")
    ax.set_ylabel("Density")
    ax.set_title("Step reward distributions (per-task -> N(0,1))")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def _draw_return(ax, returns: np.ndarray, title: str, xlabel: str) -> None:
    _kde_curve(ax, returns, _RETURN_COLOR, f"mu={returns.mean():.3f}  s2={returns.var():.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def show_plots(
    buckets: OrderedDict[str, np.ndarray],
    stats: dict[str, tuple[float, float]],
    raw_ret: np.ndarray,
    scaled_ret: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    ax_rew_raw, ax_ret_raw = axes[0]
    ax_rew_std, ax_ret_scaled = axes[1]

    _draw_reward_raw(ax_rew_raw, buckets)
    _draw_reward_std(ax_rew_std, buckets, stats)
    _draw_return(ax_ret_raw, raw_ret, "Return distribution (raw rewards)", f"G_t  (gamma={GAMMA})")
    _draw_return(ax_ret_scaled, scaled_ret, "Return distribution (scaled rewards)",
                 f"G_t  (gamma={GAMMA}, per-task standardised r_t)")

    all_axes = [ax_rew_raw, ax_rew_std, ax_ret_raw, ax_ret_scaled]
    fig.tight_layout(rect=[0, 0.07, 1, 1])  # type: ignore

    ax_check = fig.add_axes([0.01, 0.01, 0.13, 0.05])  # type: ignore
    check = CheckButtons(ax_check, ["Log Y scale"], [False])

    def _on_toggle(_label):
        scale = "log" if check.get_status()[0] else "linear"
        for ax in all_axes:
            ax.set_yscale(scale)
        fig.canvas.draw_idle()

    check.on_clicked(_on_toggle)
    plt.show()


def main():
    label = (
        MODELS_DIR / DISTILL_CKPT
        if MODEL_SOURCE == "distill"
        else f"{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}"
    )
    print("=" * 52)
    print(f"  source         {MODEL_SOURCE}")
    print(f"  model          {label}")
    print(f"  tasks          {[getattr(t, 'name', t) for t in ALLOWED_TASK_MIXING]}")
    print(f"  steps/task     {STEPS_PER_TASK:,}")
    print(f"  collect envs   {N_COLLECT_ENVS}")
    print(f"  gamma          {GAMMA}")
    print("=" * 52)

    print("\nLoading model...")
    env = make_eval_env()
    model = load_model(env)
    env.close()

    print("\nCollecting step rewards...")
    buckets, episodes = collect(model)

    print(f"\nComputing returns over {len(episodes):,} completed episodes...")
    stats = _task_stats(buckets)
    raw_ret, scaled_ret = compute_returns(episodes, stats)
    print(f"  Total return samples: {len(raw_ret):,}\n")

    print("Generating plots...")
    show_plots(buckets, stats, raw_ret, scaled_ret)
    print("\nDone.")


if __name__ == "__main__":
    main()
