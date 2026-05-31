"""
scripts/rlft/visualize_critic.py
================================

Visualize a Rudin-baseline critic's value landscape over the command space
``(cmd_vel, cmd_tilt)`` for a chosen task one-hot. Loads any SB3 PPO + RlFTPolicy
zip — by default the pretrained-critic ``final.zip`` (see pretrain_critic.py),
but a finetuned ``best/best_model.zip`` works too.

Sweeps a 2D grid for a heatmap and runs a quick simulated annealing pass to mark
the lowest-value (worst) command. Simplified from the old 3D/walk-hop version.

Run:  python scripts/rlft/visualize_critic.py
"""

import time
from gymnasium import make
import numpy as np
from stable_baselines3 import PPO

import torch
import matplotlib.pyplot as plt
from utils.paths import MODELS_DIR
from wrappers.ppo_bc.ppo_bc_env import RlFTEnv

# =========================================

# models/-relative dir + checkpoint. Defaults to the pretrained critic; point at
# rudin[_adv]/finetuned/<v> + "best/best_model" to inspect a finetuned model.
EXPERIMENT_NAME = "rudin/pretrained_critic/1.0.0"
MODEL_CHECKPOINT = "final"

STARTING_SEED: int | None = None

# task one-hot the critic conditions on (walk, flamingo, tilt). Mixes like
# (1, 0, 1) are valid too.
TASK_VEC: tuple[int, int, int] = (1, 0, 0)

# command sweep ranges (match train-time cmd_sample_range).
VEL_RANGE = (-5.0, 5.0)
TILT_RANGE = (-0.75, 0.75)

# sweep grid resolution.
N_VEL_PTS = 101
N_TILT_PTS = 61

# obs layout: [proprio(N_PROPRIO) | cmd_vel, cmd_tilt | task_id(3)].
N_PROPRIO = 14

# =========================================

TASK_NAMES: dict[tuple, str] = {
    (1, 0, 0): "walk",
    (0, 1, 0): "flamingo",
    (0, 0, 1): "tilt",
    (1, 1, 0): "walk + flamingo",
    (1, 0, 1): "walk + tilt",
    (0, 1, 1): "flamingo + tilt",
    (1, 1, 1): "walk + flamingo + tilt",
    (0, 0, 0): "idle",
}


def main():
    print("Loading environment...")
    raw = make("BipedalWalker-v3", render_mode=None)
    # manual_ctrl_mode disables the env's own cmd/task sampling so reset() gives
    # a clean base proprio snapshot without scrambling the task bits.
    wrap_env = RlFTEnv(raw, manual_ctrl_mode=True)

    print(f'Loading model "{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}"...')
    model_path = MODELS_DIR / f"{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}.zip"
    model = PPO.load(model_path, env=wrap_env, device="cpu")

    # one reset to grab a proprio snapshot; trailing 5 dims (cmd + task) ignored.
    obs, _ = wrap_env.reset(seed=STARTING_SEED)
    base_obs = obs[:N_PROPRIO].astype(np.float32)

    def V(cmd_vel: float, cmd_tilt: float) -> float:
        x = np.concatenate(
            [base_obs, np.array([cmd_vel, cmd_tilt], dtype=np.float32),
             np.asarray(TASK_VEC, dtype=np.float32)]
        )
        obs_tensor = torch.from_numpy(x).unsqueeze(0)
        sb3_policy = model.policy
        with torch.no_grad():
            critic_latent = sb3_policy.mlp_extractor.forward_critic(obs_tensor)
            value = sb3_policy.value_net(critic_latent)
        return float(value.item())

    # ---- simulated annealing over (cmd_vel, cmd_tilt) -------------------------
    SA_T0 = 1.0
    SA_ALPHA = 0.995
    SA_N_ITER = 100
    SA_STD = np.array([VEL_RANGE[1] - VEL_RANGE[0], TILT_RANGE[1] - TILT_RANGE[0]]) * 0.1
    LOW = np.array([VEL_RANGE[0], TILT_RANGE[0]])
    HIGH = np.array([VEL_RANGE[1], TILT_RANGE[1]])

    rng = np.random.default_rng(42)
    sa_current = np.array([float(rng.uniform(*VEL_RANGE)), float(rng.uniform(*TILT_RANGE))])
    sa_current_cost = V(float(sa_current[0]), float(sa_current[1]))
    sa_best = sa_current.copy()
    sa_best_cost = sa_current_cost
    sa_history: list[tuple[float, float]] = [(float(sa_current[0]), float(sa_current[1]))]

    T = SA_T0
    sa_start = time.time_ns()
    for _ in range(SA_N_ITER):
        candidate = np.clip(sa_current + rng.normal(0, SA_STD), LOW, HIGH)
        candidate_cost = V(float(candidate[0]), float(candidate[1]))
        delta = candidate_cost - sa_current_cost
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            sa_current = candidate
            sa_current_cost = candidate_cost
            if sa_current_cost < sa_best_cost:
                sa_best = sa_current.copy()
                sa_best_cost = sa_current_cost
        sa_history.append((float(sa_current[0]), float(sa_current[1])))
        T *= SA_ALPHA

    sa_dur_ms = (time.time_ns() - sa_start) / 1e6
    print(f"SA best  cmd_vel={sa_best[0]:+.4f}  cmd_tilt={sa_best[1]:+.4f}  V={sa_best_cost:.4f}  ({sa_dur_ms:.1f}ms)")

    # ---- dense grid sweep -----------------------------------------------------
    print("Sweeping landscape...")
    vel_grid = np.linspace(VEL_RANGE[0], VEL_RANGE[1], N_VEL_PTS)
    tilt_grid = np.linspace(TILT_RANGE[0], TILT_RANGE[1], N_TILT_PTS)
    # ZZ[i, j] = V(vel_grid[j], tilt_grid[i]) so row=tilt, col=vel for pcolormesh.
    ZZ = np.array([[V(float(v), float(t)) for v in vel_grid] for t in tilt_grid])

    min_idx = np.unravel_index(np.argmin(ZZ), ZZ.shape)
    grid_best_vel = float(vel_grid[min_idx[1]])
    grid_best_tilt = float(tilt_grid[min_idx[0]])
    grid_best_val = float(ZZ[min_idx])
    print(f"Grid min cmd_vel={grid_best_vel:+.4f}  cmd_tilt={grid_best_tilt:+.4f}  V={grid_best_val:.4f}")

    # ---- plot -----------------------------------------------------------------
    hist_vels = np.array([v for v, _ in sa_history])
    hist_tilts = np.array([t for _, t in sa_history])
    cmap = plt.get_cmap("coolwarm")
    point_colors = [tuple(cmap(x)) for x in np.linspace(1.0, 0.0, len(sa_history))]

    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(vel_grid, tilt_grid, ZZ, cmap="viridis", shading="auto")
    fig.colorbar(mesh, ax=ax, label="V (critic value estimate)")

    ax.scatter(
        hist_vels, hist_tilts,
        c=point_colors, s=22, edgecolor="black", linewidths=0.3, alpha=0.85,
        label="SA trajectory (red=oldest -> blue=newest)",
    )
    ax.scatter(
        [float(sa_best[0])], [float(sa_best[1])],
        color="lime", s=240, marker="*", edgecolor="black", linewidths=0.8,
        label="SA worst (min V)",
    )
    ax.scatter(
        [grid_best_vel], [grid_best_tilt],
        color="orange", s=240, marker="*", edgecolor="black", linewidths=0.8,
        label="Grid min",
    )

    task_name = TASK_NAMES.get(TASK_VEC, str(TASK_VEC))
    ax.set_xlabel("cmd_vel")
    ax.set_ylabel("cmd_tilt")
    ax.set_xlim(*VEL_RANGE)
    ax.set_ylim(*TILT_RANGE)
    ax.set_title(
        f"Critic value landscape - task = {task_name}  {TASK_VEC}\n"
        f'"{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}"'
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
