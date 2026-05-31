"""
scripts/eval/switching_and_individual.py
========================================

Parallelized evaluation of a set of policies inside the PPO_BC env (RlFTEnv),
covering ONLY two test families (future eval scripts will cover combined tasks /
everything):

  1. Task switching          — perform task A for 2 s (100 steps) then switch to
                               task B for 2 s. A switch is successful if the agent
                               never terminates. Failures are split into
                               "before switch" (fell during task A, step < 100) and
                               "after switch" (fell at/after the switch, step >= 100).
                               For after-switch failures we also record the mean
                               survival duration (frames + seconds) before termination.
  2. Individual-task success — hold a single task for 10 s (500 steps), re-sampling
                               the command every 3 s (150 steps), and measure whether
                               the agent survives the whole window.
  3. Average reward per task — collected during test 2: the env's compositional
                               reward, averaged per task (total per episode + per step).

All models consume the identical 19-dim RlFTEnv observation
``[14 proprio, cmd_vel, cmd_tilt, walk, flamingo, tilt]``; SB3 PPO, the Torch
distillation students, and the HybridModelV2 oracle all share that layout, so a
single env + obs drives every model.

Tasks are the four individual tasks. Walk is split into walk_forward / walk_backward
purely via the sign of the velocity command (0 -> forward); the observation is
unchanged. See mdp.bipedal_walker.tasks.

Parallelism: one (model, test, scope, episode-chunk) per pool job, one model + env
per worker, torch pinned to 1 thread per worker to avoid oversubscription. Defaults
to os.cpu_count() workers (override with EVAL_WORKERS); scales to any core count.

Run:  python scripts/eval/switching_and_individual.py
"""

import csv
import json
import multiprocessing as mp
import os
import warnings
from datetime import datetime

import numpy as np
import torch
from gymnasium import make
from tqdm import tqdm

from utils.paths import MODELS_DIR, rudin_distill_ckpt
from wrappers.ppo_bc.ppo_bc_env import RlFTEnv
from mdp.bipedal_walker.student import StudentModel
from mdp.bipedal_walker.hybrid import HybridModelV2

# =========================================
# config

FPS = 50

# --- tests ---
SWITCH_SEG_STEPS = 250          # 2 s per task in the switching test
INDIV_STEPS = 500               # 10 s held for the individual-task test
INDIV_MODULATE_EVERY = 150      # re-sample the command every 3 s within the task

# --- thoroughness (configurable; env vars allow quick smoke runs) ---
def _envint(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v else default

EPISODES_PER_TASK = _envint("EVAL_EPISODES_PER_TASK", 500)  # individual eps per (model, task)
EPISODES_PER_PAIR = _envint("EVAL_EPISODES_PER_PAIR", 200)  # switching eps per (model, pair)
EPISODE_CHUNK = _envint("EVAL_EPISODE_CHUNK", 100)          # eps per pool job (load balancing)

# --- env ---
EP_TIME = 30                    # >> any schedule, so truncation never pre-empts a test
USE_INDV_TASK_REW = True        # individual (single-bit) tasks need task-specific rewards

# --- parallelism ---
N_WORKERS = int(os.environ.get("EVAL_WORKERS", os.cpu_count() or 1))
PROGRESS_EVERY = 10             # episodes between progress pings

# --- reproducibility ---
SEED_BASE = 42

# --- the four individual tasks (codebase convention: walk_forward/backward/flamingo/tilt) ---
TASKS = ["walk_forward", "walk_backward", "flamingo", "tilt"]
# all ordered distinct task pairs (A -> B)
PAIRS = [(a, b) for a in TASKS for b in TASKS if a != b]

# --- models to evaluate ---
# kind: "sb3" (PPO.load), "torch" (StudentModelV2), "hybrid" (HybridModelV2 oracle).
# ref: for sb3 a ppo_bc version (timestamped dirs resolved by glob), for torch a
# (adversarial, version) tuple resolved via rudin_distill_ckpt, for hybrid unused.
MODELS = [
    dict(name="ppo_bc/2.2.1", kind="sb3", ref="2.2.1"),
    dict(name="ppo_bc/2.2.0", kind="sb3", ref="2.2.0"),
    dict(name="ppo_bc/2.0.1", kind="sb3", ref="2.0.1"),
    dict(name="ppo_bc/1.2.0", kind="sb3", ref="1.2.0"),   # predecessor
    dict(name="ppo_bc/1.0.2", kind="sb3", ref="1.0.2"),   # predecessor
    dict(name="rudin_adv/1.0.0", kind="torch", ref=(True, "1.0.0")),
    dict(name="rudin/1.0.0", kind="torch", ref=(False, "1.0.0")),
    dict(name="hybrid", kind="hybrid", ref=None),
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# =========================================


def resolve_path(spec: dict) -> str | None:
    """Resolve a model ref to a concrete checkpoint path (in the main process)."""
    if spec["kind"] == "hybrid":
        return None
    if spec["kind"] == "torch":
        adversarial, version = spec["ref"]
        p = rudin_distill_ckpt(adversarial, version, "best.pt")
        if not p.exists():
            raise FileNotFoundError(f"{spec['name']}: {p} not found")
        return str(p)
    # sb3 ppo_bc: exact dir, else a timestamped "<ver>*" dir, that has best/best_model.zip
    base = MODELS_DIR / "ppo_bc"
    exact = base / spec["ref"] / "best" / "best_model.zip"
    if exact.exists():
        return str(exact)
    for d in sorted(base.glob(f"{spec['ref']}*")):
        cand = d / "best" / "best_model.zip"
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(f"{spec['name']}: no best/best_model.zip under {base}/{spec['ref']}*")


def make_predict_fn(kind: str, path: str | None):
    """Return a uniform ``predict(obs_19) -> action_4`` for the given model kind.

    All three kinds consume the same 19-dim RlFTEnv observation."""
    if kind == "sb3":
        from stable_baselines3 import PPO

        model = PPO.load(path, device="cpu") # type: ignore

        def predict(obs):
            return model.predict(obs, deterministic=True)[0]

        return predict

    if kind == "torch":
        model = StudentModel()
        ckpt = torch.load(path, map_location="cpu", weights_only=False) # type: ignore
        model.load_state_dict(ckpt["policy"])
        model.eval()

        def predict(obs): # type: ignore
            with torch.no_grad():
                return model(torch.tensor(obs, dtype=torch.float32)).numpy()

        return predict

    if kind == "hybrid":
        model = HybridModelV2()

        def predict(obs):
            with torch.no_grad():
                return model.forward(torch.tensor(obs, dtype=torch.float32)).numpy()

        return predict

    raise ValueError(f"unknown model kind: {kind}")


def make_eval_env() -> RlFTEnv:
    """Single RlFTEnv driven manually (no internal resampling), with task-specific
    rewards enabled so individual tasks produce meaningful reward."""
    return RlFTEnv(
        make("BipedalWalker-v3", render_mode=None),
        ep_time=EP_TIME,
        use_rew_for_individual_tasks=USE_INDV_TASK_REW,
        manual_ctrl_mode=True,
    )


def sample_command(task: str, rng: np.random.Generator):
    """Return ``(task_bits, cmd_vel, cmd_tilt)`` for a task. Walk direction is the
    velocity sign (0 -> forward). Flamingo has no command; tilt commands an angle."""
    if task == "walk_forward":
        return (1, 0, 0), float(rng.uniform(0.0, 5.0)), 0.0
    if task == "walk_backward":
        return (1, 0, 0), float(rng.uniform(-5.0, 0.0)), 0.0
    if task == "flamingo":
        return (0, 1, 0), 0.0, 0.0
    if task == "tilt":
        return (0, 0, 1), 0.0, float(rng.uniform(-0.75, 0.75))
    raise ValueError(task)


def build_switch_schedule(task_a: str, task_b: str, rng: np.random.Generator):
    """[(bits, vel, tilt, n_steps)] = task A for SWITCH_SEG_STEPS then task B."""
    bits_a, vel_a, tilt_a = sample_command(task_a, rng)
    bits_b, vel_b, tilt_b = sample_command(task_b, rng)
    return [
        (bits_a, vel_a, tilt_a, SWITCH_SEG_STEPS),
        (bits_b, vel_b, tilt_b, SWITCH_SEG_STEPS),
    ]


def build_individual_schedule(task: str, rng: np.random.Generator):
    """Hold one task for INDIV_STEPS, re-sampling its command every
    INDIV_MODULATE_EVERY steps (a no-op for flamingo, which has no command)."""
    segs = []
    remaining = INDIV_STEPS
    while remaining > 0:
        n = min(INDIV_MODULATE_EVERY, remaining)
        bits, vel, tilt = sample_command(task, rng)
        segs.append((bits, vel, tilt, n))
        remaining -= n
    return segs


def run_episode(env: RlFTEnv, predict, schedule, seed: int) -> dict:
    """Drive the env through a schedule of (task_bits, cmd_vel, cmd_tilt, n_steps)
    segments. At each segment boundary the task + command are snapped and the obs
    tail rebuilt so the first action of the segment already sees the new task
    (mirrors scripts/ppo_bc/play.py). Stops on termination (a fall)."""
    np.random.seed(seed & 0x7FFFFFFF)  # RlFTEnv.reset uses global np.random for hull init
    obs, _ = env.reset(seed=seed)

    step = 0
    total_reward = 0.0
    for bits, vel, tilt, n_steps in schedule:
        env._task_id_vec = bits
        env._cmd_vec = (vel, tilt)
        env._cmd_vec_target = (vel, tilt)
        # rebuild the trailing cmd+task obs slots for the new segment
        obs = env._derive_full_obs(obs[:-5], env._effective_cmd_vec(), bits)
        for _ in range(n_steps):
            action = predict(obs)
            obs, rew, term, trunc, _ = env.step(action)
            total_reward += float(rew)
            step += 1
            if term:
                return dict(terminated=True, term_step=step, total_reward=total_reward, steps=step)
    return dict(terminated=False, term_step=None, total_reward=total_reward, steps=step)


def _empty_agg(test: str) -> dict:
    if test == "individual":
        return dict(n=0, success=0, reward_sum=0.0, steps_sum=0)
    return dict(n=0, success=0, fail_before=0, fail_after=0, after_frames_sum=0)


def _merge_agg(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def _run_job(job: dict, q) -> None:
    """Pool worker: load one model + env, run a chunk of episodes for one
    (test, scope), and emit progress + a partial aggregate via the queue."""
    torch.set_num_threads(1)
    try:
        predict = make_predict_fn(job["kind"], job["path"])
        env = make_eval_env()
        test, scope, n, base = job["test"], job["scope"], job["n_episodes"], job["seed_base"]
        agg = _empty_agg(test)

        sent = 0
        for i in range(n):
            seed = base + i
            rng = np.random.default_rng(seed)
            if test == "individual":
                sched = build_individual_schedule(scope, rng)
            else:
                sched = build_switch_schedule(scope[0], scope[1], rng)
            res = run_episode(env, predict, sched, seed)

            agg["n"] += 1
            if test == "individual":
                agg["success"] += int(not res["terminated"])
                agg["reward_sum"] += res["total_reward"]
                agg["steps_sum"] += res["steps"]
            else:
                if not res["terminated"]:
                    agg["success"] += 1
                elif res["term_step"] < SWITCH_SEG_STEPS:
                    agg["fail_before"] += 1
                else:
                    agg["fail_after"] += 1
                    agg["after_frames_sum"] += res["term_step"]

            if (i + 1) - sent >= PROGRESS_EVERY:
                q.put(("progress", (i + 1) - sent))
                sent = i + 1
        if n - sent > 0:
            q.put(("progress", n - sent))
        env.close()
        q.put(("result", job["model"], test, scope, agg))
    except Exception as e:  # surface worker errors instead of hanging the pool
        q.put(("error", job["model"], test, job.get("scope"), repr(e)))


def _build_jobs(entries):
    """Fan each (model, test, scope) out into EPISODE_CHUNK-sized jobs."""
    jobs = []
    for e in entries:
        for task in TASKS:
            for off in range(0, EPISODES_PER_TASK, EPISODE_CHUNK):
                n = min(EPISODE_CHUNK, EPISODES_PER_TASK - off)
                jobs.append(dict(model=e["name"], kind=e["kind"], path=e["path"],
                                 test="individual", scope=task,
                                 n_episodes=n, seed_base=SEED_BASE + off))
        for pair in PAIRS:
            for off in range(0, EPISODES_PER_PAIR, EPISODE_CHUNK):
                n = min(EPISODE_CHUNK, EPISODES_PER_PAIR - off)
                jobs.append(dict(model=e["name"], kind=e["kind"], path=e["path"],
                                 test="switching", scope=pair,
                                 n_episodes=n, seed_base=SEED_BASE + 100000 + off))
    return jobs


# ---- result assembly + reporting ----------------------------------------------


def _finalize(entries, indiv, switch):
    """Turn raw aggregates into reportable metrics per model."""
    out = {}
    for e in entries:
        name = e["name"]
        # individual per task
        ind = {}
        succ_vals, rew_vals = [], []
        for task in TASKS:
            a = indiv[name][task]
            n = max(a["n"], 1)
            sr = a["success"] / n
            avg_total = a["reward_sum"] / n
            avg_ps = a["reward_sum"] / max(a["steps_sum"], 1)
            ind[task] = dict(n=a["n"], success_rate=sr,
                             avg_total_reward=avg_total, avg_per_step_reward=avg_ps)
            succ_vals.append(sr)
            rew_vals.append(avg_total)
        # switching overall (sum across pairs) + per pair
        tot = dict(n=0, success=0, fail_before=0, fail_after=0, after_frames_sum=0)
        per_pair = {}
        for pair in PAIRS:
            a = switch[name][pair]
            _merge_agg(tot, a)
            n = max(a["n"], 1)
            fails_p = a["fail_before"] + a["fail_after"]
            after_f = (a["after_frames_sum"] / a["fail_after"]) if a["fail_after"] else 0.0
            per_pair[f"{pair[0]}->{pair[1]}"] = dict(
                n=a["n"], success_rate=a["success"] / n,
                fail_before=a["fail_before"], fail_after=a["fail_after"],
                pct_fail_before=(a["fail_before"] / fails_p) if fails_p else 0.0,
                pct_fail_after=(a["fail_after"] / fails_p) if fails_p else 0.0,
                after_survival_frames=after_f,
                after_survival_seconds=after_f / FPS,
            )
        n = max(tot["n"], 1)
        fails = tot["fail_before"] + tot["fail_after"]
        after_frames = (tot["after_frames_sum"] / tot["fail_after"]) if tot["fail_after"] else 0.0
        sw = dict(
            n=tot["n"],
            success_rate=tot["success"] / n,
            fail_before=tot["fail_before"],
            fail_after=tot["fail_after"],
            pct_fail_before=(tot["fail_before"] / fails) if fails else 0.0,
            pct_fail_after=(tot["fail_after"] / fails) if fails else 0.0,
            after_survival_frames=after_frames,
            after_survival_seconds=after_frames / FPS,
            per_pair=per_pair,
        )
        out[name] = dict(
            individual=ind,
            individual_overall_success=float(np.mean(succ_vals)),
            individual_overall_reward=float(np.mean(rew_vals)),
            switching=sw,
        )
    return out


def _print_report(entries, results):
    names = [e["name"] for e in entries]
    nw = max(len(n) for n in names)

    def hdr(title):
        print(f"\n{title}\n{'=' * (len(title))}")

    hdr("Individual-task success rate (%)")
    cols = TASKS + ["overall"]
    print(f"{'model':<{nw}}  " + "  ".join(f"{c:>14}" for c in cols))
    for n in names:
        r = results[n]
        cells = [f"{r['individual'][t]['success_rate']*100:>13.1f}%" for t in TASKS]
        cells.append(f"{r['individual_overall_success']*100:>13.1f}%")
        print(f"{n:<{nw}}  " + "  ".join(cells))

    hdr("Average reward per task (total episode reward)")
    print(f"{'model':<{nw}}  " + "  ".join(f"{c:>14}" for c in TASKS))
    for n in names:
        r = results[n]
        cells = [f"{r['individual'][t]['avg_total_reward']:>14.1f}" for t in TASKS]
        print(f"{n:<{nw}}  " + "  ".join(cells))

    hdr("Task switching")
    cols = ["success", "%fail_before", "%fail_after", "after_surv(s)", "after_surv(f)"]
    print(f"{'model':<{nw}}  " + "  ".join(f"{c:>14}" for c in cols))
    for n in names:
        s = results[n]["switching"]
        cells = [
            f"{s['success_rate']*100:>13.1f}%",
            f"{s['pct_fail_before']*100:>13.1f}%",
            f"{s['pct_fail_after']*100:>13.1f}%",
            f"{s['after_survival_seconds']:>14.2f}",
            f"{s['after_survival_frames']:>14.1f}",
        ]
        print(f"{n:<{nw}}  " + "  ".join(cells))


def _write_outputs(entries, results, meta):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    base = os.path.join(OUTPUT_DIR, f"switching_and_individual_{stamp}")
    written = []

    with open(base + ".json", "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    written.append(base + ".json")

    # 1) SUMMARY — wide, one row per model (at-a-glance comparison)
    fields = ["model"]
    for t in TASKS:
        fields += [f"ind_succ_{t}", f"ind_reward_{t}", f"ind_reward_ps_{t}"]
    fields += [
        "ind_succ_overall",
        "switch_succ", "switch_pct_fail_before", "switch_pct_fail_after",
        "switch_after_surv_frames", "switch_after_surv_sec",
    ]
    with open(base + "_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in entries:
            n = e["name"]
            r = results[n]
            row = {"model": n}
            for t in TASKS:
                it = r["individual"][t]
                row[f"ind_succ_{t}"] = round(it["success_rate"], 4)
                row[f"ind_reward_{t}"] = round(it["avg_total_reward"], 3)
                row[f"ind_reward_ps_{t}"] = round(it["avg_per_step_reward"], 4)
            s = r["switching"]
            row["ind_succ_overall"] = round(r["individual_overall_success"], 4)
            row["switch_succ"] = round(s["success_rate"], 4)
            row["switch_pct_fail_before"] = round(s["pct_fail_before"], 4)
            row["switch_pct_fail_after"] = round(s["pct_fail_after"], 4)
            row["switch_after_surv_frames"] = round(s["after_survival_frames"], 2)
            row["switch_after_surv_sec"] = round(s["after_survival_seconds"], 3)
            w.writerow(row)
    written.append(base + "_summary.csv")

    # 2) INDIVIDUAL detail — one row per (model, task)
    with open(base + "_individual_by_task.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "n", "success_rate",
                    "avg_total_reward", "avg_per_step_reward"])
        for e in entries:
            n = e["name"]
            for t in TASKS:
                it = results[n]["individual"][t]
                w.writerow([n, t, it["n"], round(it["success_rate"], 4),
                            round(it["avg_total_reward"], 3), round(it["avg_per_step_reward"], 4)])
    written.append(base + "_individual_by_task.csv")

    # 3) SWITCHING detail — one row per (model, ordered pair A->B)
    with open(base + "_switching_by_pair.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "pair", "from", "to", "n", "success_rate",
                    "fail_before", "fail_after", "pct_fail_before", "pct_fail_after",
                    "after_surv_frames", "after_surv_sec"])
        for e in entries:
            n = e["name"]
            per_pair = results[n]["switching"]["per_pair"]
            for a, b in PAIRS:
                d = per_pair[f"{a}->{b}"]
                w.writerow([n, f"{a}->{b}", a, b, d["n"], round(d["success_rate"], 4),
                            d["fail_before"], d["fail_after"],
                            round(d["pct_fail_before"], 4), round(d["pct_fail_after"], 4),
                            round(d["after_survival_frames"], 2), round(d["after_survival_seconds"], 3)])
    written.append(base + "_switching_by_pair.csv")

    print("\nWrote:")
    for p in written:
        print(f"  {p}")


def main():
    # optional model filter (comma-separated names) for quick subsets / smoke runs
    only = os.environ.get("EVAL_MODELS")
    specs = MODELS
    if only:
        wanted = {s.strip() for s in only.split(",") if s.strip()}
        specs = [m for m in MODELS if m["name"] in wanted]

    # resolve checkpoint paths up front (fail fast on missing models)
    entries = []
    for spec in specs:
        try:
            path = resolve_path(spec)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue
        entries.append(dict(name=spec["name"], kind=spec["kind"], path=path))

    jobs = _build_jobs(entries)
    total_episodes = sum(j["n_episodes"] for j in jobs)
    n_jobs = len(jobs)

    # masters: results[model][task] / [pair]
    indiv = {e["name"]: {t: _empty_agg("individual") for t in TASKS} for e in entries}
    switch = {e["name"]: {p: _empty_agg("switching") for p in PAIRS} for e in entries}

    print(f"Models:    {len(entries)}  ({', '.join(e['name'] for e in entries)})")
    print(f"Tasks:     {len(TASKS)}  ({', '.join(TASKS)})")
    print(f"Switching: {len(PAIRS)} ordered pairs x {EPISODES_PER_PAIR} eps")
    print(f"Individual:{len(TASKS)} tasks x {EPISODES_PER_TASK} eps "
          f"({INDIV_STEPS} steps, modulate every {INDIV_MODULATE_EVERY})")
    print(f"Jobs:      {n_jobs}   workers: {N_WORKERS}   total episodes: {total_episodes:,}")

    completed = 0
    errors = []
    with mp.Manager() as manager:
        q = manager.Queue()
        with mp.Pool(processes=N_WORKERS) as pool:
            for job in jobs:
                pool.apply_async(_run_job, args=(job, q))
            pool.close()

            with tqdm(total=total_episodes, desc="Evaluating", unit="ep") as pbar:
                while completed < n_jobs:
                    msg = q.get()
                    if msg[0] == "progress":
                        pbar.update(msg[1])
                    elif msg[0] == "error":
                        _, model, test, scope, err = msg
                        errors.append((model, test, scope, err))
                        completed += 1
                        pbar.set_postfix(done=f"{completed}/{n_jobs}", errors=len(errors))
                    else:
                        _, model, test, scope, agg = msg
                        bucket = indiv if test == "individual" else switch
                        _merge_agg(bucket[model][scope], agg)
                        completed += 1
                        pbar.set_postfix(done=f"{completed}/{n_jobs}", errors=len(errors))
            pool.join()

    if errors:
        print(f"\n!! {len(errors)} job(s) errored:")
        for model, test, scope, err in errors[:20]:
            print(f"   {model} {test} {scope}: {err}")

    results = _finalize(entries, indiv, switch)
    _print_report(entries, results)

    meta = dict(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        episodes_per_task=EPISODES_PER_TASK,
        episodes_per_pair=EPISODES_PER_PAIR,
        switch_seg_steps=SWITCH_SEG_STEPS,
        indiv_steps=INDIV_STEPS,
        indiv_modulate_every=INDIV_MODULATE_EVERY,
        fps=FPS,
        tasks=TASKS,
        n_workers=N_WORKERS,
        seed_base=SEED_BASE,
        models=[e["name"] for e in entries],
        errors=[f"{m} {t} {s}: {err}" for m, t, s, err in errors],
    )
    _write_outputs(entries, results, meta)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="__array_wrap__ must accept context and return_scalar arguments",
        category=DeprecationWarning,
    )
    main()
