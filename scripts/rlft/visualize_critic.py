import os
import time
from gymnasium import make
import numpy as np
import pygame
from stable_baselines3 import PPO

from pynput import keyboard
from pynput.keyboard import Key, KeyCode

import torch
import matplotlib.pyplot as plt
from utils.paths import MODELS_DIR
from wrappers.plot_env import Plotter
from wrappers.plot_reward_env import RewardPlotter
from wrappers.bipedal_walker.hop_env import HopEnv
from wrappers.bipedal_walker.hop_finetune_env import HopFTEnv
from wrappers.bipedal_walker.walk_env import WalkEnv
from wrappers.bipedal_walker.walk_finetune_env import WalkFTEnv
from wrappers.bipedal_walker.rltf_env import RlFTEnv
from wrappers.bipedal_walker.proprio_wrapper import ProprioObsWrapper

# =========================================

EXPERIMENT_NAME = "rlft/finetuned/ml_3.4.3_g97-02_54_26-2026_05_13"
MODEL_CHECKPOINT = "best/best_model"

STARTING_SEED = None

# =========================================

def main():
    # load env
    print("Loading environments...")
    raw = make("BipedalWalker-v3", render_mode=None)

    rlft_env = RlFTEnv(
        raw,
        vel_switching_freq=2,
        task_switching_freq=5,
        vel_interp_speed=5.0,
    )
    wrap_env = rlft_env

    # pygame.init()
    # screen = pygame.display.set_mode((600, 400))
    # clock = pygame.time.Clock()

    # load model
    print(f'Loading model "{MODEL_CHECKPOINT}"...')
    model_path = MODELS_DIR / f"{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}.zip"
    model = PPO.load(model_path, env=wrap_env, device="cpu")

    def do_reset(cmd_vel: float):
        obs, _ = wrap_env.reset(seed=STARTING_SEED)
        rlft_env._cmd_vel = cmd_vel
        rlft_env._cmd_vel_target = cmd_vel
        rlft_env._cmd_task_id = 0
        obs = rlft_env._derive_full_obs(obs[:-3], cmd_vel, 0)
        return obs

    obs = do_reset(0)  # get an initial observation first
    print(obs.shape)
    
    # extract out base data [0, 14]
    base_obs = obs[:14]

    def V(base_obs: np.ndarray, cmd: np.ndarray) -> float:
        sb3_policy = model.policy
        obs_tensor = torch.tensor(np.concatenate([base_obs, cmd]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            critic_latent = sb3_policy.mlp_extractor.forward_critic(obs_tensor)
            value = sb3_policy.value_net(critic_latent)
        return float(value.item())

    # task ∈ [0,1] continuous → one-hot at evaluation time: <0.5 → walk [1,0], >=0.5 → hop [0,1]
    def objective(cmd_vel: float, task: float) -> float:
        task_spec = np.array([1.0, 0.0]) if task < 0.5 else np.array([0.0, 1.0])
        return V(base_obs, np.array([cmd_vel, *task_spec]))

    # simulated annealing over (cmd_vel, task) jointly
    SA_T0 = 1.0
    SA_ALPHA = 0.995
    SA_N_ITER = 100
    SA_STD = np.array([5.0, 0.15])  # per-dimension step sizes

    rng = np.random.default_rng(42)
    sa_current = np.array([float(rng.uniform(-5.0, 5.0)), float(rng.uniform(0.0, 1.0))])
    sa_current_cost = objective(float(sa_current[0]), float(sa_current[1]))
    sa_best = sa_current.copy()
    sa_best_cost = sa_current_cost
    sa_history: list[tuple[float, float, float]] = [
        (float(sa_current[0]), float(sa_current[1]), sa_current_cost)
    ]

    T = SA_T0
    sa_start_time = time.time_ns()
    for _ in range(SA_N_ITER):
        candidate = np.clip(sa_current + rng.normal(0, SA_STD), [-5.0, 0.0], [5.0, 1.0])
        candidate_cost = objective(float(candidate[0]), float(candidate[1]))
        delta = candidate_cost - sa_current_cost
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            sa_current = candidate
            sa_current_cost = candidate_cost
            if sa_current_cost < sa_best_cost:
                sa_best = sa_current.copy()
                sa_best_cost = sa_current_cost
        sa_history.append((float(sa_current[0]), float(sa_current[1]), sa_current_cost))
        T *= SA_ALPHA

    best_task_name = "walk" if sa_best[1] < 0.5 else "hop"
    print(f"SA finished in {((time.time_ns() - sa_start_time) / 1e+6):.2f}ms with {SA_N_ITER} iterations")
    print(f"SA best  cmd_vel={sa_best[0]:.4f}  task={sa_best[1]:.4f} ({best_task_name})  V={sa_best_cost:.4f}")

    # 2D landscape sweep: cmd_vel × task
    print("Sweeping landscape...")
    vel_grid = np.arange(-5.0, 5.001, 0.1)
    task_grid = np.linspace(0.0, 1.0, 20)
    VV, TT = np.meshgrid(vel_grid, task_grid)
    ZZ = np.array([[objective(float(v), float(t)) for v in vel_grid] for t in task_grid])

    min_idx = np.unravel_index(np.argmin(ZZ), ZZ.shape)
    grid_best_vel = float(vel_grid[min_idx[1]])
    grid_best_task = float(task_grid[min_idx[0]])
    grid_best_val = float(ZZ[min_idx])
    grid_best_task_name = "walk" if grid_best_task < 0.5 else "hop"
    print(f"Grid min  cmd_vel={grid_best_vel:.4f}  task={grid_best_task:.4f} ({grid_best_task_name})  V={grid_best_val:.4f}")

    # plot 3D interactive surface
    hist_vels = np.array([v for v, _, _ in sa_history])
    hist_tasks = np.array([t for _, t, _ in sa_history])
    hist_vals = np.array([c for _, _, c in sa_history])
    n_pts = len(sa_history)

    # oldest → red (coolwarm=1.0), newest → blue (coolwarm=0.0)
    cmap = plt.get_cmap("coolwarm")
    point_colors = [tuple(cmap(x)) for x in np.linspace(1.0, 0.0, n_pts)]

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(VV, TT, ZZ, alpha=0.5, cmap="viridis")
    ax.scatter(hist_vels, hist_tasks, hist_vals, color=point_colors, s=20, alpha=0.8,  # type: ignore[arg-type]
               label="SA trajectory")
    ax.scatter([float(sa_best[0])], [float(sa_best[1])], [sa_best_cost],  # type: ignore[arg-type]
               color="green", s=200, marker="*",
               label=f"SA best  cmd_vel={sa_best[0]:.3f}  task={best_task_name}  V={sa_best_cost:.3f}")
    ax.scatter([grid_best_vel], [grid_best_task], [grid_best_val],  # type: ignore[arg-type]
               color="orange", s=200, marker="*",
               label=f"Grid min  cmd_vel={grid_best_vel:.3f}  task={grid_best_task_name}  V={grid_best_val:.3f}")

    ax.set_xlabel("cmd_vel")
    ax.set_ylabel("task (0=walk, 1=hop)")
    ax.set_zlabel("V (critic value estimate)")  # type: ignore[attr-defined]
    ax.set_title("Critic Value Landscape  (red=oldest SA point, blue=newest)")
    ax.legend()

    # replace matplotlib's default trackball (which introduces roll) with
    # a simple yaw/pitch controller: left-drag horizontal → azimuth, vertical → elevation
    ax.disable_mouse_rotation()  # type: ignore[attr-defined]
    _drag: dict = {}

    def _on_press(event):
        if event.inaxes is ax and event.button == 1:
            _drag["x0"] = event.x
            _drag["y0"] = event.y
            _drag["azim0"] = ax.azim
            _drag["elev0"] = ax.elev

    def _on_release(event):
        if event.button == 1:
            _drag.clear()

    def _on_move(event):
        if "x0" not in _drag:
            return
        dx = event.x - _drag["x0"]
        dy = event.y - _drag["y0"]
        ax.azim = _drag["azim0"] - dx * 0.4
        ax.elev = float(np.clip(_drag["elev0"] - dy * 0.4, -90, 90))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_press)
    fig.canvas.mpl_connect("button_release_event", _on_release)
    fig.canvas.mpl_connect("motion_notify_event", _on_move)

    plt.tight_layout()
    plt.show()

    return

    # manually render
    def render():
        frame = wrap_env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))  # type: ignore
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        clock.tick(50)

    while 1:
        pygame.event.pump()  # keep window alive on pause

        if _sim_res:
            # print out total rewards before resetting
            print(f"Total rewards: {total_rewards}")
            total_rewards = 0

            _sim_res = False
            # obs = do_reset()
            render()
            continue

        # if _sim_paused:
            if not _sim_step:
                continue
            else:
                _sim_step = False

        assert wrap_env.action_space.shape is not None

        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, term, trunc, _ = wrap_env.step(action)
        total_rewards += float(rewards)


if __name__ == "__main__":
    main()


