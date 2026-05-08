"""
Evaluate a distilled or finetuned model on RlFTEnv and plot:
  - Per-task step reward distributions (walk vs. hop)
  - Combined discounted return distribution, before and after per-task reward scaling

2×2 figure layout:
  (0,0)  Raw step reward distributions     (walk, hop, combined)
  (1,0)  Standardised reward distributions (each task → N(0,1), combined)
  (0,1)  Combined return distribution       from raw rewards
  (1,1)  Combined return distribution       after per-task reward standardisation

Returns are computed over full episode sequences — G_t = r_t + γ·G_{t+1} — so
task-switching within an episode is handled correctly. The "after scaling" returns
are produced by standardising each r_t by the statistics of its task before
computing G_t, showing what the critic's target landscape looks like under
per-task reward normalisation.

Interactive controls:
  Toolbar  — zoom, pan, home (built-in matplotlib buttons)
  Checkbox — toggle log Y scale on all four subplots simultaneously
"""

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from scipy.stats import gaussian_kde
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import tqdm
from pathlib import Path

from utils.paths import MODELS_DIR
from wrappers.bipedal_walker.rltf_env import RlFTEnv
from mdp.bipedal_walker.rlft_policy import RlFTPolicy, _MODEL_CONFIGS

# =========================================
# Configuration — edit here
# =========================================

# "distill" → build a PPO shell, load raw actor weights from a distilled .pt
# "ppo"     → load a fully-trained PPO .zip (finetuned or pretrained critic)
MODEL_SOURCE = "distill"

# Used when MODEL_SOURCE == "distill"
MODEL_SIZE = "ml"

# Used when MODEL_SOURCE == "ppo"
EXPERIMENT_NAME  = "rlft/finetuned/ml_1-16_09_36-2026_05_05"
MODEL_CHECKPOINT = "best/best_model"

# Step reward samples to gather for each task independently.
# Completed episodes are tracked throughout; returns are computed from those.
STEPS_PER_TASK = 1_000_000

# Parallel envs for collection
N_COLLECT_ENVS = 14 * 4

# Discount factor for return computation
GAMMA = 0.9

# RlFTEnv settings
EP_TIME          = 10
VEL_SAMPLE_RANGE = (0, 5)
VEL_SAMPLE_ZERO  = 0.2
VEL_INTERP_SPEED = 0.5

# =========================================


def make_eval_env() -> RlFTEnv:
    base = gym.make("BipedalWalker-v3")
    return RlFTEnv(
        base,
        ep_time=EP_TIME,
        vel_sample_range=VEL_SAMPLE_RANGE,
        vel_sample_zero=VEL_SAMPLE_ZERO,
        vel_interp_speed=VEL_INTERP_SPEED,
    )


def load_model(env: RlFTEnv) -> PPO:
    if MODEL_SOURCE == "distill":
        hidden_dims = _MODEL_CONFIGS[MODEL_SIZE]
        model = PPO(
            RlFTPolicy,
            env=env,
            policy_kwargs=dict(hidden_dims=hidden_dims, activation_fn=torch.nn.ELU),
        )
        student_path = MODELS_DIR / f"distill/{MODEL_SIZE}/best.pt"
        print(f"  Loading distilled weights: {student_path}")
        student_sd = torch.load(
            student_path, map_location="cpu", weights_only=False
        )["policy"]
        layer_idx = [int(k.split(".")[1]) for k in list(student_sd.keys())[::2]]

        mlp_ext    = model.policy.mlp_extractor
        action_net = model.policy.action_net
        with torch.no_grad():
            for idx in layer_idx[:-1]:
                mlp_ext.policy_net[idx].weight.copy_(student_sd[f"policy.{idx}.weight"])  # type: ignore
                mlp_ext.policy_net[idx].bias.copy_(student_sd[f"policy.{idx}.bias"])  # type: ignore
            action_net.weight.copy_(student_sd[f"policy.{layer_idx[-1]}.weight"])  # type: ignore
            action_net.bias.copy_(student_sd[f"policy.{layer_idx[-1]}.bias"])  # type: ignore
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


# Type alias for a completed episode: list of (reward, task_id) pairs
Episode = list[tuple[float, int]]


def collect(model: PPO) -> tuple[np.ndarray, np.ndarray, list[Episode]]:
    """
    Roll out across N_COLLECT_ENVS parallel envs, maintaining full episode
    sequences. Stops once STEPS_PER_TASK step rewards have been gathered for
    each task. All episodes completed by that point are returned for return
    computation.

    Returns: (walk_rew, hop_rew, completed_episodes)
    """
    vec_env = SubprocVecEnv([make_eval_env] * N_COLLECT_ENVS)

    walk_rew: list[float] = []
    hop_rew:  list[float] = []

    ep_bufs: list[Episode] = [[] for _ in range(N_COLLECT_ENVS)]
    completed: list[Episode] = []

    obs = vec_env.reset()
    pbar = tqdm(total=STEPS_PER_TASK * 2, unit="steps", desc="Collecting")
    prev = 0

    while len(walk_rew) < STEPS_PER_TASK or len(hop_rew) < STEPS_PER_TASK:
        actions, _ = model.predict(obs, deterministic=True)  # type: ignore
        obs, rewards, dones, infos = vec_env.step(actions)

        for i in range(N_COLLECT_ENVS):
            task = infos[i]["task"]
            r    = float(rewards[i])

            if task == 0 and len(walk_rew) < STEPS_PER_TASK:
                walk_rew.append(r)
            elif task == 1 and len(hop_rew) < STEPS_PER_TASK:
                hop_rew.append(r)

            ep_bufs[i].append((r, task))

            if dones[i]:
                completed.append(ep_bufs[i])
                ep_bufs[i] = []

        curr = min(len(walk_rew), STEPS_PER_TASK) + min(len(hop_rew), STEPS_PER_TASK)
        pbar.update(curr - prev)
        prev = curr

    pbar.close()
    vec_env.close()
    return (
        np.array(walk_rew[:STEPS_PER_TASK]),
        np.array(hop_rew[:STEPS_PER_TASK]),
        completed,
    )


def compute_returns(
    episodes: list[Episode],
    mu_w: float, std_w: float,
    mu_h: float, std_h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute two return arrays from the collected episodes:
      raw_returns    — G_t from unmodified rewards
      scaled_returns — G_t after each r_t is standardised by its task stats
    """
    raw_all:    list[float] = []
    scaled_all: list[float] = []

    for ep in tqdm(episodes):
        raw_rews = [r for r, _ in ep]
        scaled_rews = [
            (r - mu_w) / (std_w + 1e-8) if t == 0 else (r - mu_h) / (std_h + 1e-8)
            for r, t in ep
        ]
        raw_all.extend(_discounted_returns(raw_rews))
        scaled_all.extend(_discounted_returns(scaled_rews))

    return np.array(raw_all), np.array(scaled_all)


# ── plotting ──────────────────────────────────────────────────────────────────

_WALK_COLOR   = "#2196F3"
_HOP_COLOR    = "#FF5722"
_TOTAL_COLOR  = "#4CAF50"
_RETURN_COLOR = "#9C27B0"


def _kde_curve(ax, data: np.ndarray, color: str, label: str):
    kde = gaussian_kde(data)
    x   = np.linspace(data.min(), data.max(), 2000)
    y   = kde(x)
    ax.plot(x, y, color=color, linewidth=2, label=label)
    ax.fill_between(x, y, alpha=0.12, color=color)


def _draw_reward_raw(
    ax, walk: np.ndarray, hop: np.ndarray
) -> tuple[float, float, float, float]:
    total = np.concatenate([walk, hop])
    mu_w, var_w = walk.mean(),  walk.var()
    mu_h, var_h = hop.mean(),   hop.var()
    mu_t, var_t = total.mean(), total.var()

    _kde_curve(ax, walk,  _WALK_COLOR,  f"Walk   μ={mu_w:.3f}  σ²={var_w:.3f}")
    _kde_curve(ax, hop,   _HOP_COLOR,   f"Hop    μ={mu_h:.3f}  σ²={var_h:.3f}")
    _kde_curve(ax, total, _TOTAL_COLOR, f"Total  μ={mu_t:.3f}  σ²={var_t:.3f}")

    ax.set_xlabel("Step reward", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Step reward distributions (raw)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    print("  Rewards (raw)")
    print(f"    Walk  — μ={mu_w:.4f}  σ²={var_w:.4f}  σ={np.sqrt(var_w):.4f}")
    print(f"    Hop   — μ={mu_h:.4f}  σ²={var_h:.4f}  σ={np.sqrt(var_h):.4f}")
    print(f"    Total — μ={mu_t:.4f}  σ²={var_t:.4f}  σ={np.sqrt(var_t):.4f}")

    return float(mu_w), float(np.sqrt(var_w)), float(mu_h), float(np.sqrt(var_h))


def _draw_reward_std(
    ax, walk: np.ndarray, hop: np.ndarray,
    mu_w: float, std_w: float, mu_h: float, std_h: float,
) -> None:
    walk_s   = (walk - mu_w) / (std_w + 1e-8)
    hop_s    = (hop  - mu_h) / (std_h + 1e-8)
    combined = np.concatenate([walk_s, hop_s])
    mu_c, var_c = combined.mean(), combined.var()

    _kde_curve(ax, walk_s,   _WALK_COLOR,  "Walk (standardised)")
    _kde_curve(ax, hop_s,    _HOP_COLOR,   "Hop (standardised)")
    _kde_curve(ax, combined, _TOTAL_COLOR, f"Combined  μ={mu_c:.3f}  σ²={var_c:.3f}")

    ax.set_xlabel("Standardised step reward", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Step reward distributions (per-task → N(0,1))", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def _draw_return(ax, returns: np.ndarray, title: str, xlabel: str) -> None:
    mu, var = returns.mean(), returns.var()
    _kde_curve(ax, returns, _RETURN_COLOR, f"μ={mu:.3f}  σ²={var:.3f}")

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    print(f"  {title}")
    print(f"    μ={mu:.4f}  σ²={var:.4f}  σ={np.sqrt(var):.4f}")


def show_plots(
    walk_rew: np.ndarray, hop_rew: np.ndarray,
    raw_ret: np.ndarray, scaled_ret: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    ax_rew_raw, ax_ret_raw    = axes[0]
    ax_rew_std, ax_ret_scaled = axes[1]

    print("Computing raw reward KDE...")
    mu_w, std_w, mu_h, std_h = _draw_reward_raw(ax_rew_raw, walk_rew, hop_rew)
    print("Computing standardized reward KDE...")
    _draw_reward_std(ax_rew_std, walk_rew, hop_rew, mu_w, std_w, mu_h, std_h)
    print("Computing raw return KDE...")
    _draw_return(ax_ret_raw,    raw_ret,    "Return distribution (raw rewards)",    f"G_t  (γ={GAMMA})")
    print("Computing scaled return KDE...")
    _draw_return(ax_ret_scaled, scaled_ret, "Return distribution (scaled rewards)", f"G_t  (γ={GAMMA}, per-task standardised r_t)")

    all_axes = [ax_rew_raw, ax_rew_std, ax_ret_raw, ax_ret_scaled]
    fig.tight_layout(rect=[0, 0.07, 1, 1])  # type: ignore

    # ── log Y checkbox ────────────────────────────────────────────────────────
    ax_check = fig.add_axes([0.01, 0.01, 0.13, 0.05])  # type: ignore
    check    = CheckButtons(ax_check, ["Log Y scale"], [False])

    def _on_toggle(_label):
        scale = "log" if check.get_status()[0] else "linear"
        for ax in all_axes:
            ax.set_yscale(scale)
        fig.canvas.draw_idle()

    check.on_clicked(_on_toggle)
    plt.show()


def main():
    label = (
        f"distill/{MODEL_SIZE}/best.pt"
        if MODEL_SOURCE == "distill"
        else f"{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}"
    )
    print("=" * 52)
    print(f"  source         {MODEL_SOURCE}")
    print(f"  model          {label}")
    print(f"  steps/task     {STEPS_PER_TASK:,}")
    print(f"  collect envs   {N_COLLECT_ENVS}")
    print(f"  gamma          {GAMMA}")
    print("=" * 52)

    print("\nLoading model...")
    env   = make_eval_env()
    model = load_model(env)
    env.close()

    print(f"\nCollecting {STEPS_PER_TASK:,} step rewards per task ({N_COLLECT_ENVS} parallel envs)...")
    walk_rew, hop_rew, episodes = collect(model)

    print(f"\nComputing returns over {len(episodes):,} completed episodes...")
    mu_w = float(walk_rew.mean())
    std_w = float(walk_rew.std())
    mu_h = float(hop_rew.mean())
    std_h = float(hop_rew.std())
    raw_ret, scaled_ret = compute_returns(episodes, mu_w, std_w, mu_h, std_h)

    print(f"  Total return samples: {len(raw_ret):,}\n")
    print("\nGenerating plots...")
    show_plots(walk_rew, hop_rew, raw_ret, scaled_ret)

    print("\nDone.")


if __name__ == "__main__":
    main()
