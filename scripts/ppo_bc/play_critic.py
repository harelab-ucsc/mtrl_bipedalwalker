"""
Play the trained PPO_BC policy with random cmd sampling, alongside a live
critic value landscape.

Layout: matplotlib figure with two subplots — left = V(cmd_vel, cmd_tilt)
heatmap for the fixed TASK_VEC; right = bipedal walker rgb_array frame.

Controls:
    space  toggle play / pause          (start state: paused)
    s      single step (while paused)
    r      reset env + recompute landscape
    q      quit

Cmd_vel and cmd_tilt are sampled by the env itself (random, not adversarial).
The SA red star marks where the critic *would* assign the lowest value under
the current proprio — purely a visualization, the env does not follow it.
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

EXPERIMENT_NAME = "ppo_bc/critic_pretrain/1.2.0"
MODEL_CHECKPOINT = "final"
STARTING_SEED: int | None = None

# task one-hot the env + policy run under. (walk, flamingo, tilt).
TASK_VEC: tuple[int, int, int] = (1, 0, 1)

# cmd sweep + sampling ranges (matches train-time CMD_SAMPLE_RANGE).
VEL_RANGE  = (-5.0, 5.0)
TILT_RANGE = (-0.75, 0.75)

# env params (mirror scripts/ppo_bc/train.py defaults)
EP_TIME              = 10
CMD_SWITCHING_TIME   = (3.0, 4.0)
TASK_SWITCHING_TIME  = 6.0
CMD_INTERP_SPEED     = (5.0, 1.0)
CMD_SAMPLE_ZERO      = (0.2, 0.15)
HULL_X_RANGE         = (20.0, 60.0)

# heatmap grid (kept moderate so recompute on reset stays sub-second).
N_VEL_PTS  = 80
N_TILT_PTS = 40

# SA hyperparams for the adversarial-worst marker.
SA_T0     = 1.0
SA_ALPHA  = 0.995
SA_N_ITER = 100
SA_STD    = np.array(
    [(VEL_RANGE[1] - VEL_RANGE[0]) * 0.5, (TILT_RANGE[1] - TILT_RANGE[0]) * 0.5]
)

N_PROPRIO = 14
FPS       = 50  # target loop rate

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

# matplotlib defaults grab 's', 'r', 'q' for save/home/quit shortcuts — clear
# those so our key handler sees them.
plt.rcParams["keymap.save"] = []
plt.rcParams["keymap.home"] = []
plt.rcParams["keymap.quit"] = []


def main():
    task_name = TASK_NAMES.get(TASK_VEC, str(TASK_VEC))

    print("Loading environments...")
    raw = make("BipedalWalker-v3", render_mode="rgb_array")

    # Task pinned to TASK_VEC by handing the env a single-element allowed list.
    # cmd_vel + cmd_tilt are sampled by the env's own scheduler (random).
    env = RlFTEnv(
        raw,
        ep_time=EP_TIME,
        cmd_switching_time=CMD_SWITCHING_TIME,
        task_switching_time=TASK_SWITCHING_TIME,
        cmd_interp_speed=CMD_INTERP_SPEED,
        cmd_sample_range=(VEL_RANGE, TILT_RANGE),
        cmd_sample_zero=CMD_SAMPLE_ZERO,
        allowed_task_mixing=[TASK_VEC],
        # single forced task: switching must allow repeats (the without-replacement
        # default can't draw 2 distinct tasks from a 1-task list and would raise at init).
        task_switch_replacement=True,
        hull_x_range=HULL_X_RANGE,
        manual_ctrl_mode=False,
    )

    print(f'Loading model "{MODEL_CHECKPOINT}"...')
    model_path = MODELS_DIR / f"{EXPERIMENT_NAME}/{MODEL_CHECKPOINT}.zip"
    model = PPO.load(model_path, env=env, device="cpu")
    sb3_policy = model.policy

    obs, _ = env.reset(seed=STARTING_SEED)
    task_arr_np = np.asarray(TASK_VEC, dtype=np.float32)
    # mask matching RlFTEnv._effective_cmd_vec: walk flag gates vel, tilt flag
    # gates tilt — flamingo alone zeros both. Apply this everywhere a cmd is
    # fed to the critic, so the visualization reflects what the policy actually
    # sees rather than the raw sampled cmd.
    mask_arr_np = np.array([TASK_VEC[0], TASK_VEC[2]], dtype=np.float32)

    def _mask_cmd(cmd_vel: float, cmd_tilt: float) -> tuple[float, float]:
        return float(cmd_vel * mask_arr_np[0]), float(cmd_tilt * mask_arr_np[1])

    def V(base_obs: np.ndarray, cmd_vel: float, cmd_tilt: float) -> float:
        mv, mt = _mask_cmd(cmd_vel, cmd_tilt)
        x = np.concatenate(
            [base_obs, np.array([mv, mt], dtype=np.float32), task_arr_np]
        ).astype(np.float32)
        with torch.no_grad():
            obs_t = torch.from_numpy(x).unsqueeze(0)
            critic_latent = sb3_policy.mlp_extractor.forward_critic(obs_t)
            return float(sb3_policy.value_net(critic_latent).item())

    def compute_landscape(base_obs: np.ndarray):
        """Batched V over the full (cmd_vel, cmd_tilt) grid in a single forward."""
        vel_grid  = np.linspace(VEL_RANGE[0],  VEL_RANGE[1],  N_VEL_PTS).astype(np.float32)
        tilt_grid = np.linspace(TILT_RANGE[0], TILT_RANGE[1], N_TILT_PTS).astype(np.float32)
        VV, TT = np.meshgrid(vel_grid, tilt_grid)
        n = VV.size
        # mask the swept cmds the same way the env masks live cmds. Dims gated
        # off by the task collapse to 0, so the heatmap is flat along them.
        cmds = np.stack([VV.ravel(), TT.ravel()], axis=-1) * mask_arr_np
        full = np.concatenate(
            [
                np.tile(base_obs, (n, 1)),
                cmds,
                np.tile(task_arr_np, (n, 1)),
            ],
            axis=-1,
        ).astype(np.float32)
        with torch.no_grad():
            obs_t = torch.from_numpy(full)
            critic_latent = sb3_policy.mlp_extractor.forward_critic(obs_t)
            values = sb3_policy.value_net(critic_latent).squeeze(-1).cpu().numpy()
        return vel_grid, tilt_grid, values.reshape(VV.shape)

    def run_sa(base_obs: np.ndarray) -> tuple[float, float, float]:
        LOW  = np.array([VEL_RANGE[0],  TILT_RANGE[0]])
        HIGH = np.array([VEL_RANGE[1],  TILT_RANGE[1]])
        rng = np.random.default_rng(42)
        cur = np.array(
            [float(rng.uniform(*VEL_RANGE)), float(rng.uniform(*TILT_RANGE))]
        )
        cur_cost = V(base_obs, float(cur[0]), float(cur[1]))
        best, best_cost = cur.copy(), cur_cost
        T = SA_T0
        for _ in range(SA_N_ITER):
            cand = np.clip(cur + rng.normal(0, SA_STD), LOW, HIGH)
            cand_cost = V(base_obs, float(cand[0]), float(cand[1]))
            delta = cand_cost - cur_cost
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                cur, cur_cost = cand, cand_cost
                if cur_cost < best_cost:
                    best, best_cost = cur.copy(), cur_cost
            T *= SA_ALPHA
        return float(best[0]), float(best[1]), best_cost

    # ---- initial landscape ----
    print("Computing initial landscape...")
    base_obs = obs[:N_PROPRIO].astype(np.float32)
    _, _, ZZ = compute_landscape(base_obs)
    sa_vel, sa_tilt, sa_val = run_sa(base_obs)

    # ---- figure ----
    # constrained_layout handles the legend-outside-axes case automatically.
    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    fig.canvas.manager.set_window_title("PPO_BC — play + critic landscape")  # type: ignore[union-attr]
    ax_heat   = fig.add_subplot(1, 2, 1)
    ax_walker = fig.add_subplot(1, 2, 2)
    ax_walker.axis("off")

    # imshow instead of pcolormesh — uploads a single texture per redraw
    # instead of rasterizing N_VEL_PTS*N_TILT_PTS quads. Order-of-magnitude
    # faster when the surface refreshes every frame.
    mesh = ax_heat.imshow(
        ZZ, cmap="plasma", origin="lower",
        extent=(VEL_RANGE[0], VEL_RANGE[1], TILT_RANGE[0], TILT_RANGE[1]),
        aspect="auto", interpolation="bicubic",
    )
    fig.colorbar(mesh, ax=ax_heat, label="V (critic value)")

    adv_marker = ax_heat.scatter(
        [sa_vel], [sa_tilt],
        color="red", s=280, marker="*", edgecolor="black", linewidths=0.9,
        label="adversarial worst (SA)",
    )
    # current cmd marker shows the masked (effective) cmd — i.e. what the
    # policy actually sees once the task gates are applied.
    eff_vel0, eff_tilt0 = env._effective_cmd_vec()
    cur_marker = ax_heat.scatter(
        [eff_vel0], [eff_tilt0],
        color="cyan", s=200, marker="o", edgecolor="black", linewidths=0.9,
        label="current env cmd (masked)",
    )

    # leave breathing room around the data so the SA star is visible when it
    # lands on an edge of the cmd range.
    vel_pad  = (VEL_RANGE[1]  - VEL_RANGE[0])  * 0.06
    tilt_pad = (TILT_RANGE[1] - TILT_RANGE[0]) * 0.06

    ax_heat.set_xlabel("cmd_vel")
    ax_heat.set_ylabel("cmd_tilt")
    ax_heat.set_xlim(VEL_RANGE[0]  - vel_pad,  VEL_RANGE[1]  + vel_pad)
    ax_heat.set_ylim(TILT_RANGE[0] - tilt_pad, TILT_RANGE[1] + tilt_pad)
    ax_heat.set_title(f"Critic value — task = {task_name}  {TASK_VEC}")
    # legend outside the axes (below the plot) so it never occludes the
    # heatmap markers, even at the edges.
    ax_heat.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.10),
        ncol=2, framealpha=0.95, fontsize=9,
    )

    frame = env.render()
    if frame is None:
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
    walker_img = ax_walker.imshow(frame)
    ax_walker.set_title("bipedal walker")

    stats_text = ax_heat.text(
        0.02, 0.98, "",
        transform=ax_heat.transAxes, ha="left", va="top",
        family="monospace", fontsize=9, color="black",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white", alpha=0.85, edgecolor="0.3",
        ),
    )

    # mutable state poked by the keyboard handler.
    state = {
        "paused":  False,
        "step":    False,
        "reset":   False,
        "quit":    False,
    }

    def on_key(event):
        k = event.key
        if k == " ":
            state["paused"] = not state["paused"]
            print("Paused" if state["paused"] else "Resumed")
        elif k == "s":
            state["step"] = True
        elif k == "r":
            state["reset"] = True
        elif k == "q":
            state["quit"] = True

    def on_close(_event):
        state["quit"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    # running max for the colorbar — only ratchets upward across steps so the
    # heatmap shows magnitude relative to the best value ever seen, not just
    # the best value in the current frame. Reset on env reset.
    cmap_state: dict[str, float | None] = {"max": None}

    def refresh_landscape():
        nonlocal ZZ, sa_vel, sa_tilt, sa_val
        base = obs[:N_PROPRIO].astype(np.float32)
        _, _, ZZ = compute_landscape(base)
        sa_vel, sa_tilt, sa_val = run_sa(base)
        mesh.set_data(ZZ)
        zmin, zmax = float(ZZ.min()), float(ZZ.max())
        prev_max = cmap_state["max"]
        if prev_max is not None:
            zmax = max(zmax, prev_max)
        cmap_state["max"] = zmax
        if zmax - zmin < 1e-9:
            zmax = zmin + 1e-9  # avoid degenerate colorbar when ZZ is flat
        mesh.set_clim(zmin, zmax)
        # show the SA result projected onto the dims the task actually uses,
        # so it lands on the meaningful subspace of the heatmap.
        sa_mv, sa_mt = _mask_cmd(sa_vel, sa_tilt)
        adv_marker.set_offsets([[sa_mv, sa_mt]])

    def update_stats():
        # report the masked (effective) cmd — same convention the marker uses.
        cv, ct = env._effective_cmd_vec()
        sa_mv, sa_mt = _mask_cmd(sa_vel, sa_tilt)
        stats_text.set_text(
            f"task     : {task_name} {TASK_VEC}\n"
            f"cur cmd  : vel={cv:+.3f}  tilt={ct:+.3f}  (masked)\n"
            f"adv worst: vel={sa_mv:+.3f}  tilt={sa_mt:+.3f}  V={sa_val:.3f}\n"
            f"state    : {'PAUSED' if state['paused'] else 'PLAYING'}\n"
            f"keys     : space=play/pause  s=step  r=reset  q=quit"
        )

    update_stats()
    plt.show(block=False)
    # one explicit draw to settle the constrained layout (legend + colorbar
    # positions), then freeze the layout engine. Otherwise constrained_layout
    # re-runs its solver on every redraw, which dominates frame time.
    fig.canvas.draw()
    fig.canvas.flush_events()
    try:
        fig.set_layout_engine("none")
    except Exception:
        # older matplotlib: fall back to the deprecated attribute toggle.
        fig.set_constrained_layout(False)  # type: ignore[attr-defined]

    print(f'=== Starting experiment "{EXPERIMENT_NAME}" ===')
    print(f"task = {task_name} {TASK_VEC}  (start state: {'PAUSED' if state['paused'] else 'PLAYING'})")
    print("Controls: space=play/pause  s=step  r=reset  q=quit")

    # ---- main loop ----
    while not state["quit"]:
        loop_start = time.time()

        do_step = (not state["paused"]) or state["step"]
        state["step"] = False

        if do_step:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                state["reset"] = True
            refresh_landscape()

        if state["reset"]:
            obs, _ = env.reset()
            print("Recomputing landscape after reset...")
            cmap_state["max"] = None
            refresh_landscape()
            state["reset"] = False

        # tick the walker frame + cmd marker every loop iteration so the UI
        # feels live (especially the cmd interp dot) even while paused.
        frame = env.render()
        if frame is not None:
            walker_img.set_data(frame)
        eff_vel, eff_tilt = env._effective_cmd_vec()
        cur_marker.set_offsets([[eff_vel, eff_tilt]])
        update_stats()

        # queue a redraw and process GUI events. start_event_loop also runs
        # the paint, so we don't need a separate plt.pause (which would call
        # draw_idle a second time).
        fig.canvas.draw_idle()
        elapsed = time.time() - loop_start
        fig.canvas.start_event_loop(max(1e-3, 1.0 / FPS - elapsed))

    plt.close(fig)


if __name__ == "__main__":
    main()
