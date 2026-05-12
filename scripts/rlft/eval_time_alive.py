import argparse
import multiprocessing as mp
import warnings
from collections import Counter

import numpy as np
import torch
from gymnasium import make
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import tqdm

from utils.paths import MODELS_DIR
from wrappers.bipedal_walker.rltf_env import RlFTEnv
from mdp.bipedal_walker.hybrid import HybridModel

T_EVAL = 5_000_000       # total env-steps per agent
UPDATE_INTERVAL = 1_000  # env-steps between progress-queue sends
N_ENVS = 4               # parallel env instances per agent (override with --n-envs)

# cmd_vel (obs[14]) is the raw velocity command in m/s from vel_sample_range=(-5, 5).
# It is NOT normalized. This threshold separates slow/stopped from directional motion.
VEL_DIR_THRESH = 0.5  # m/s

# (display_name, path_relative_to_MODELS_DIR, agent_type)
# agent_type "ppo"    → load via PPO.load()
# agent_type "hybrid" → oracle router between the 4 expert PPO models (path unused)
AGENTS: list[tuple[str, str, str]] = [
    # --- Finetuned models ---
    ("ml_3.3.1a_g97", "rlft/finetuned/ml_3.3.1a_g97-15_16_31-2026_05_11/best/best_model", "ppo"),
    ("ml_3.3.1_g97",  "rlft/finetuned/ml_3.3.1_g97-15_16_22-2026_05_11/best/best_model",  "ppo"),
    ("ml_3.2.8_g97",  "rlft/finetuned/ml_3.2.8_g97-19_04_21-2026_05_08/best/best_model",  "ppo"),
    ("xl_3.2.8_g97",  "rlft/finetuned/xl_3.2.8_g97-19_04_25-2026_05_08/best/best_model",  "ppo"),
    # ("ml_3.2.7_g97",  "rlft/finetuned/ml_3.2.7_g97-15_52_42-2026_05_08/best/best_model",  "ppo"),
    # ("ml_3.2.6_g97",  "rlft/finetuned/ml_3.2.6_g97-15_21_59-2026_05_08/best/best_model",  "ppo"),
    # ("ml_3.2.5_g97",  "rlft/finetuned/ml_3.2.5_g97-15_00_01-2026_05_08/best/best_model",  "ppo"),
    # ("ml_3.2.4_g95",  "rlft/finetuned/ml_3.2.4_g95-13_48_36-2026_05_08/best/best_model",  "ppo"),
    # ("ml_3.2.3_g95",  "rlft/finetuned/ml_3.2.3_g95-13_26_23-2026_05_08/best/best_model",  "ppo"),
    # ("ml_3.2.2_g99",  "rlft/finetuned/ml_3.2.2_g99-20_41_51-2026_05_07/best/best_model",  "ppo"),
    # ("ml_3.2.1_g99",  "rlft/finetuned/ml_3.2.1_g99-18_32_27-2026_05_07/best/best_model",  "ppo"),
    # ("ml_3.2_g99",    "rlft/finetuned/ml_3.2_g99-17_57_41-2026_05_07/best/best_model",    "ppo"),
    # ("ml_3.1_g99",    "rlft/finetuned/ml_3.1_g99-17_38_43-2026_05_07/best/best_model",    "ppo"),
    # ("ml_2_g95",      "rlft/finetuned/ml_2_g95-02_20_25-2026_05_07/best/best_model",      "ppo"),
    # ("ml_1",          "rlft/finetuned/ml_1-16_09_36-2026_05_05/best/best_model",          "ppo"),
    # --- Hybrid baseline ---
    ("hybrid",        "",                                                                 "hybrid"),
    # --- Pretrained (no fine-tuning) ---
    # ("ml_3.2.8 (pre)", "rlft/pretrain/ml_3.2.8_g97/best_model",     "ppo"),
    # ("xl_3.2.8a (pre)","rlft/pretrain/xl_3.2.8a_g97/best_model",    "ppo"),
    # ("m (pre)",        "rlft/pretrain/m/best/best_model",            "ppo"),
    # ("l (pre)",        "rlft/pretrain/l/best/best_model",            "ppo"),
    # ("xl (pre)",       "rlft/pretrain/xl/best/best_model",           "ppo"),
]


def _vel_dir(cmd_vel: float) -> str:
    # cmd_vel is obs[14]: raw m/s command from RlFTEnv, range (-5, 5), NOT normalized.
    # VEL_DIR_THRESH=0.5 m/s separates slow/stopped commands from directional ones.
    if cmd_vel > VEL_DIR_THRESH:
        return "fwd"
    if cmd_vel < -VEL_DIR_THRESH:
        return "bwd"
    return "stop"


def _seg_label(task_id: int, cmd_vel: float) -> str:
    return f"{'walk' if task_id == 0 else 'hop'}_{_vel_dir(cmd_vel)}"


def _make_env():
    # Module-level so SubprocVecEnv can pickle it for subprocess workers.
    return RlFTEnv(
        make("BipedalWalker-v3", render_mode=None),
        ep_time=20,
        vel_switching_freq=3,
        task_switching_freq=6,
        vel_interp_speed=3.0,
    )


def _eval_agent(name: str, model_path: str, agent_type: str, q, n_envs: int) -> None:
    warnings.filterwarnings(
        "ignore",
        message="__array_wrap__ must accept context and return_scalar arguments",
        category=DeprecationWarning,
    )

    env = SubprocVecEnv([_make_env for _ in range(n_envs)])

    if agent_type == "ppo":
        model = PPO.load(str(MODELS_DIR / model_path), env=env, device="cpu")

        def predict(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action
    else:
        hybrid = HybridModel()

        def predict(obs):
            # obs shape: (n_envs, obs_dim) — loop since HybridModel expects single obs
            return np.array([
                hybrid.forward(torch.tensor(obs[i], dtype=torch.float32)).numpy()
                for i in range(n_envs)
            ])

    time_alive:    list[float]                      = []
    seg_alive:     dict[str, list[float]]           = {}
    # failure_modes entries: (at_label, from_label | None)
    # Recorded only on true termination (fall), not on timeout truncation.
    # at_label   = (task, direction) the agent was in when it fell,
    #              e.g. "hop_bwd" = commanded to hop while going backward
    # from_label = (task, direction) before the most recent task/direction switch,
    #              or None if the agent never switched during this episode.
    # Together these reveal transitions like "fell during hop_bwd right after
    # switching from walk_fwd", pointing to instability at command transitions.
    failure_modes: list[tuple[str, str | None]] = []

    # Per-env episode state.
    # alive[i]     counts steps survived in current episode (incremented post non-done step).
    # seg_count[i] counts steps in current command segment (same cadence as alive[i]).
    alive     = np.zeros(n_envs, dtype=int)
    seg_count = np.zeros(n_envs, dtype=int)

    reset_result = env.reset()
    obs: np.ndarray = np.asarray(reset_result[0] if isinstance(reset_result, tuple) else reset_result)

    # obs[i][15] > 0.5 → walk (task 0); obs[i][14] = cmd_vel
    seg_labels:  list[str]            = [_seg_label(0 if obs[i][15] > 0.5 else 1, float(obs[i][14])) for i in range(n_envs)]
    prev_labels: list[str | None]     = [None] * n_envs

    n_steps   = T_EVAL // n_envs
    last_sent = 0

    for step in range(n_steps):
        actions = predict(obs)
        # SubprocVecEnv auto-resets done envs; obs[i] after done is the new episode's first obs.
        obs_raw, _, dones, infos = env.step(actions)
        obs = np.asarray(obs_raw)

        for i in range(n_envs):
            done = bool(dones[i])
            info = infos[i]

            if done:
                # "TimeLimit.truncated" is True when the episode ended by timeout (trunc),
                # absent or False when the hull hit the ground (term).
                trunc = info.get("TimeLimit.truncated", False)

                # +1: the step that caused done counts as a step survived.
                time_alive.append(alive[i] + 1)

                if not trunc:
                    failure_modes.append((seg_labels[i], prev_labels[i]))

                # obs[i] is already the reset obs for the new episode.
                seg_labels[i]  = _seg_label(0 if obs[i][15] > 0.5 else 1, float(obs[i][14]))
                prev_labels[i] = None
                alive[i]       = 0
                seg_count[i]   = 0
            else:
                alive[i]     += 1
                seg_count[i] += 1

                # obs[i] is the post-step obs. Detect task or direction switch mid-episode.
                # obs[i][14] = cmd_vel (m/s, post-switch); obs[i][15:17] = one-hot task.
                task_id       = 0 if obs[i][15] > 0.5 else 1
                current_label = _seg_label(task_id, float(obs[i][14]))
                if current_label != seg_labels[i]:
                    seg_alive.setdefault(seg_labels[i], []).append(seg_count[i])
                    prev_labels[i] = seg_labels[i]
                    seg_labels[i]  = current_label
                    seg_count[i]   = 0

        total_steps = (step + 1) * n_envs
        if total_steps - last_sent >= UPDATE_INTERVAL:
            q.put(total_steps - last_sent)
            last_sent = total_steps

    remaining = n_steps * n_envs - last_sent
    if remaining > 0:
        q.put(remaining)

    # Close open episodes and segments for each env.
    for i in range(n_envs):
        if alive[i] > 0:
            time_alive.append(alive[i])
        if seg_count[i] > 0:
            seg_alive.setdefault(seg_labels[i], []).append(seg_count[i])

    env.close()

    avg_alive      = float(np.mean(time_alive)) if time_alive else 0.0
    seg_avgs       = {lbl: float(np.mean(ts)) for lbl, ts in seg_alive.items() if ts}
    best_label     = max(seg_avgs, key=seg_avgs.__getitem__) if seg_avgs else "n/a"
    best_label_avg = seg_avgs.get(best_label, 0.0)

    top5: list[str] = []
    if failure_modes:
        counter = Counter(failure_modes)
        total   = len(failure_modes)
        for (at, frm), cnt in counter.most_common(5):
            frm_str = frm if frm is not None else "—"
            top5.append(f"{at} ← {frm_str} ({cnt / total * 100:.0f}%)")

    q.put(("result", name, {
        "avg_alive":      avg_alive,
        "seg_avgs":       seg_avgs,
        "best_label":     best_label,
        "best_label_avg": best_label_avg,
        "top5_failures":  top5,
        "n_falls":        len(failure_modes),
    }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-envs", type=int, default=N_ENVS,
        help=f"parallel env instances per agent (default: {N_ENVS})",
    )
    args   = parser.parse_args()
    n_envs = args.n_envs

    agents = [(n, p, t) for n, p, t in AGENTS
              if t == "hybrid" or (MODELS_DIR / p).with_suffix(".zip").exists()]

    n_agents    = len(agents)
    total_steps = n_agents * T_EVAL

    with mp.Manager() as manager:
        q = manager.Queue()

        with mp.Pool() as pool:
            for name, path, atype in agents:
                pool.apply_async(_eval_agent, args=(name, path, atype, q, n_envs))
            pool.close()

            results: dict[str, dict] = {}
            completed = 0

            with tqdm(total=total_steps, desc="Evaluating", unit="step", unit_scale=True) as pbar:
                while completed < n_agents:
                    msg = q.get()
                    if isinstance(msg, int):
                        pbar.update(msg)
                    else:
                        _, name, data = msg
                        results[name] = data
                        completed += 1
                        pbar.set_postfix(done=f"{completed}/{n_agents} agents")

            pool.join()

    rows = sorted(
        [(name, results[name]) for name, _, _ in agents if name in results],
        key=lambda r: r[1]["avg_alive"],
        reverse=True,
    )

    agent_names = ", ".join(n for n, _, _ in agents)
    print(f"\nEval conditions:")
    print(f"  agents:        {n_agents}  ({agent_names})")
    print(f"  steps / agent: {T_EVAL:,}")
    print(f"  n_envs:        {n_envs}")
    print(f"  total steps:   {total_steps:,}")

    name_w    = max(len(n) for n, _ in rows)
    alive_w   = 10
    seg_lbl_w = 12
    seg_avg_w = 10

    header = "  ".join([
        f"{'Model':<{name_w}}",
        f"{'Avg Alive':>{alive_w}}",
        f"{'Best Seg':<{seg_lbl_w}}",
        f"{'Seg Avg':>{seg_avg_w}}",
    ])
    sep = "=" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, data in rows:
        print("  ".join([
            f"{name:<{name_w}}",
            f"{data['avg_alive']:>{alive_w}.1f}",
            f"{data['best_label']:<{seg_lbl_w}}",
            f"{data['best_label_avg']:>{seg_avg_w}.1f}",
        ]))
    print(sep)

    print("\n--- Failure modes (falls only, top 5 by frequency) ---")
    print("     Format: <fell_during> ← <switched_from>  (— = no prior switch in episode)")
    for name, data in rows:
        print(f"\n{name}  [{data['n_falls']} falls]:")
        if data["top5_failures"]:
            for rank, entry in enumerate(data["top5_failures"], 1):
                print(f"  {rank}. {entry}")
        else:
            print("  (no falls recorded — all episodes ended by timeout)")


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="__array_wrap__ must accept context and return_scalar arguments",
        category=DeprecationWarning,
    )
    main()
