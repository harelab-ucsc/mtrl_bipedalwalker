import multiprocessing as mp
import subprocess
import time
import warnings
from collections import Counter

import numpy as np
import torch
from gymnasium import make
from stable_baselines3 import PPO
from tqdm import tqdm

from utils.paths import MODELS_DIR
from utils.logging import fmt_duration
from wrappers.bipedal_walker.rltf_env import RlFTEnv
from mdp.bipedal_walker.hybrid import HybridModel

T_EVAL        = 5_000_000  # total env-steps per agent
UPDATE_INTERVAL = 1_000    # env-steps between progress-queue sends
TOTAL_THREADS = 14*4         # total parallel workers spread across all agents

# cmd_vel (obs[14]) is the raw velocity command in m/s from vel_sample_range=(-5, 5).
# It is NOT normalized. This threshold separates slow/stopped from directional motion.
VEL_DIR_THRESH = 0.5  # m/s

# (display_name, path_relative_to_MODELS_DIR, agent_type)
# agent_type "ppo"    → load via PPO.load()
# agent_type "hybrid" → oracle router between the 4 expert PPO models (path unused)
AGENTS: list[tuple[str, str, str]] = [
    # --- Finetuned models ---
    ("ml_3.4.1_g97",  "rlft/finetuned/ml_3.4.1_g97-02_53_58-2026_05_13/best/best_model",  "ppo"),
    ("ml_3.4.2_g97",  "rlft/finetuned/ml_3.4.2_g99-02_54_12-2026_05_13/best/best_model",  "ppo"),
    ("ml_3.4.3_g97",  "rlft/finetuned/ml_3.4.3_g97-02_54_26-2026_05_13/best/best_model",  "ppo"),
    ("ml_3.4.4_g97",  "rlft/finetuned/ml_3.4.4_g97-02_54_53-2026_05_13/best/best_model",  "ppo"),
    
    ("ml_3.3.2a_g97", "rlft/finetuned/ml_3.3.2a_g97-01_08_38-2026_05_12/best/best_model", "ppo"),
    ("ml_3.3.2_g97",  "rlft/finetuned/ml_3.3.2_g97-01_08_13-2026_05_12/best/best_model",  "ppo"),
    ("ml_3.3.1a_g97", "rlft/finetuned/ml_3.3.1a_g97-15_16_31-2026_05_11/best/best_model", "ppo"),
    ("ml_3.3.1_g97",  "rlft/finetuned/ml_3.3.1_g97-15_16_22-2026_05_11/best/best_model",  "ppo"),
    ("ml_3.2.8_g97",  "rlft/finetuned/ml_3.2.8_g97-19_04_21-2026_05_08/best/best_model",  "ppo"),
    ("xl_3.2.8_g97",  "rlft/finetuned/xl_3.2.8_g97-19_04_25-2026_05_08/best/best_model",  "ppo"),
    
    # ("ml_3.3.5_g97",  "rlft/finetuned/ml_3.3.5_g97-22_25_13-2026_05_12/best/best_model",  "ppo"),
    # ("ml_3.3.5a_g97", "rlft/finetuned/ml_3.3.5a_g97-22_27_26-2026_05_12/best/best_model", "ppo"),
    
    # ("ml_3.3.4_g97",  "rlft/finetuned/ml_3.3.4_g97-14_11_25-2026_05_12/best/best_model",  "ppo"),
    # ("ml_3.3.4a_g97", "rlft/finetuned/ml_3.3.4a_g97-14_11_39-2026_05_12/best/best_model", "ppo"),
    # ("ml_3.3.4_g95",  "rlft/finetuned/ml_3.3.4_g95-14_12_09-2026_05_12/best/best_model",  "ppo"),
    # ("ml_3.3.4a_g95", "rlft/finetuned/ml_3.3.4a_g95-14_12_26-2026_05_12/best/best_model", "ppo"),
    
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
    if cmd_vel > VEL_DIR_THRESH:
        return "fwd"
    if cmd_vel < -VEL_DIR_THRESH:
        return "bwd"
    return "stop"


def _seg_label(task_id: int, cmd_vel: float) -> str:
    return f"{'walk' if task_id == 0 else 'hop'}_{_vel_dir(cmd_vel)}"


def _eval_agent_worker(name: str, model_path: str, agent_type: str, n_steps: int, q) -> None:
    warnings.filterwarnings(
        "ignore",
        message="__array_wrap__ must accept context and return_scalar arguments",
        category=DeprecationWarning,
    )

    env = RlFTEnv(
        make("BipedalWalker-v3", render_mode=None),
        ep_time=20,
        vel_switching_freq=3,
        task_switching_freq=6,
        vel_interp_speed=6.0,
    )

    if agent_type == "ppo":
        model = PPO.load(str(MODELS_DIR / model_path), env=env, device="cpu")

        def predict(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action
    else:
        hybrid = HybridModel()

        def predict(obs):
            return hybrid.forward(torch.tensor(obs, dtype=torch.float32)).numpy()

    time_alive:    list[float]                      = []
    seg_alive:     dict[str, list[float]]           = {}
    # failure_modes entries: (at_label, from_label | None)
    # Recorded only on true termination (fall), not on timeout truncation.
    # at_label   = (task, direction) the agent was in when it fell
    # from_label = (task, direction) before the most recent task/direction switch,
    #              or None if the agent never switched during this episode.
    failure_modes: list[tuple[str, str | None]] = []

    obs, info = env.reset()
    seg_label: str = _seg_label(info["task"], float(obs[14]))
    prev_label: str | None = None
    alive = 0
    seg_count = 0
    done = False
    last_sent = 0

    for step in range(n_steps):
        if done:
            time_alive.append(alive)
            obs, info = env.reset()
            seg_label  = _seg_label(info["task"], float(obs[14]))
            prev_label = None
            alive      = 0
            seg_count  = 0
            done       = False
        else:
            alive     += 1
            seg_count += 1

        action = predict(obs)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc

        # Detect task or direction switch mid-episode.
        # obs[14] = cmd_vel (m/s, post-switch value set by RlFTEnv.step).
        # obs[15:17] = one-hot task embedding (walk→[1,0], hop→[0,1]).
        task_id       = 0 if obs[15] > 0.5 else 1
        current_label = _seg_label(task_id, float(obs[14]))
        if current_label != seg_label:
            seg_alive.setdefault(seg_label, []).append(seg_count)
            prev_label = seg_label
            seg_label  = current_label
            seg_count  = 0

        if term:
            # Record failure mode on true falls only (term=True means hull hit ground;
            # trunc=True means episode timed out — not a failure worth counting).
            failure_modes.append((seg_label, prev_label))

        if (step + 1) % UPDATE_INTERVAL == 0:
            q.put(UPDATE_INTERVAL)
            last_sent = step + 1

    remaining = n_steps - last_sent
    if remaining > 0:
        q.put(remaining)

    # Close the last open episode and segment
    time_alive.append(alive)
    if seg_count > 0:
        seg_alive.setdefault(seg_label, []).append(seg_count)

    env.close()

    q.put(("partial_result", name, {
        "time_alive":    time_alive,
        "seg_alive":     seg_alive,
        "failure_modes": failure_modes,
    }))


def _aggregate_results(partials: list[dict]) -> dict:
    all_time_alive:    list[float]                  = []
    all_seg_alive:     dict[str, list[float]]       = {}
    all_failure_modes: list[tuple[str, str | None]] = []

    for p in partials:
        all_time_alive.extend(p["time_alive"])
        for lbl, times in p["seg_alive"].items():
            all_seg_alive.setdefault(lbl, []).extend(times)
        all_failure_modes.extend(p["failure_modes"])

    avg_alive      = float(np.mean(all_time_alive)) if all_time_alive else 0.0
    seg_avgs       = {lbl: float(np.mean(ts)) for lbl, ts in all_seg_alive.items() if ts}
    best_label     = max(seg_avgs, key=seg_avgs.__getitem__) if seg_avgs else "n/a"
    best_label_avg = seg_avgs.get(best_label, 0.0)

    top5: list[str] = []
    if all_failure_modes:
        counter = Counter(all_failure_modes)
        total   = len(all_failure_modes)
        for (at, frm), cnt in counter.most_common(5):
            frm_str = frm if frm is not None else "—"
            top5.append(f"{at} ← {frm_str} ({cnt / total * 100:.0f}%)")

    return {
        "avg_alive":      avg_alive,
        "seg_avgs":       seg_avgs,
        "best_label":     best_label,
        "best_label_avg": best_label_avg,
        "top5_failures":  top5,
        "n_falls":        len(all_failure_modes),
    }


def main():
    t0 = time.time()

    print(f"[{time.time() - t0:.2f}s] Resolving agents...")
    agents = [(n, p, t) for n, p, t in AGENTS
              if t == "hybrid" or (MODELS_DIR / p).with_suffix(".zip").exists()]

    n_agents    = len(agents)
    total_steps = n_agents * T_EVAL

    # Distribute TOTAL_THREADS workers across agents as evenly as possible.
    # Each agent always gets at least 1 worker; extra slots are distributed
    # round-robin. Each worker runs its own single env.
    base_workers = max(1, TOTAL_THREADS // n_agents)
    extra        = max(0, TOTAL_THREADS - base_workers * n_agents)

    tasks: list[tuple[str, str, str, int]] = []
    workers_per_agent: dict[str, int] = {}
    for i, (name, path, atype) in enumerate(agents):
        n_workers      = base_workers + (1 if i < extra else 0)
        workers_per_agent[name] = n_workers
        base_steps     = T_EVAL // n_workers
        step_remainder = T_EVAL % n_workers
        for j in range(n_workers):
            steps = base_steps + (1 if j < step_remainder else 0)
            tasks.append((name, path, atype, steps))

    n_pool_workers = sum(workers_per_agent.values())

    workers_summary = ", ".join(f"{n}×{workers_per_agent[n]}" for n, _, _ in agents)
    print(f"[{time.time() - t0:.2f}s] Dispatching {len(tasks)} tasks across {n_pool_workers} workers  ({workers_summary})")

    with mp.Manager() as manager:
        q = manager.Queue()

        with mp.Pool(processes=n_pool_workers) as pool:
            for name, path, atype, steps in tasks:
                pool.apply_async(_eval_agent_worker, args=(name, path, atype, steps, q))
            pool.close()

            results:         dict[str, dict]       = {}
            partial_results: dict[str, list[dict]] = {name: [] for name, _, _ in agents}
            pending:         dict[str, int]        = dict(workers_per_agent)
            completed = 0

            with tqdm(total=total_steps, desc="Evaluating", unit="step", unit_scale=True) as pbar:
                while completed < n_agents:
                    msg = q.get()
                    if isinstance(msg, int):
                        pbar.update(msg)
                    elif msg[0] == "partial_result":
                        _, name, data = msg
                        partial_results[name].append(data)
                        pending[name] -= 1
                        if pending[name] == 0:
                            results[name] = _aggregate_results(partial_results[name])
                            completed += 1
                            pbar.set_postfix(done=f"{completed}/{n_agents} agents")
                            tqdm.write(f"[{time.time() - t0:.2f}s] Agent done: {name}  "
                                       f"(avg_alive={results[name]['avg_alive']:.1f})")

            pool.join()

    duration = fmt_duration(time.time() - t0)
    print(f"[{time.time() - t0:.2f}s] Eval complete. Total time: {duration}")
    subprocess.run(
        [
            "osascript", "-e",
            f'display notification "Finished in {duration}" with title "Eval complete"',
        ],
        check=False,
    )

    rows = sorted(
        [(name, results[name]) for name, _, _ in agents if name in results],
        key=lambda r: r[1]["avg_alive"],
        reverse=True,
    )

    agent_names = ", ".join(n for n, _, _ in agents)
    print(f"\nEval conditions:")
    print(f"  agents:          {n_agents}  ({agent_names})")
    print(f"  steps / agent:   {T_EVAL:,}")
    print(f"  total steps:     {total_steps:,}")
    print(f"  total threads:   {TOTAL_THREADS}  "
          f"({base_workers}–{base_workers + (1 if extra else 0)} workers/agent)")

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
