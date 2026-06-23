#!/usr/bin/env python3
"""
scripts/eval/run_bc_ablation_evals.py
=====================================

Run the six BC-ablation eval configs back to back, one at a time, streaming each
eval's live progress to the terminal:

    single      -> bc_ablation_single_task_best / _final
    combination -> bc_ablation_combination_best / _final
    switching   -> bc_ablation_switching_best   / _final

Discord notifications (best-effort, never crash the run):
  - one when the batch starts,
  - an hourly heartbeat while an eval is running (``--heartbeat-s`` to change),
  - one each time an eval finishes (or fails),
  - a final @-ping (mention) once all six are done.

The webhook + mention id are reused from the ablation sweep config
(scripts/ppo_bc_ablation/sweep.yaml -> ``notify`` block) so the channel has a
single source of truth. If that file is missing the notifier falls back to a
local macOS banner.

    python scripts/eval/run_bc_ablation_evals.py
    python scripts/eval/run_bc_ablation_evals.py --heartbeat-s 1800
    python scripts/eval/run_bc_ablation_evals.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parents[1]
SWEEP_YAML = ROOT / "scripts" / "ppo_bc_ablation" / "sweep.yaml"

# reuse the stdlib-only notifier from the ablation sweep (webhook lives in sweep.yaml)
sys.path.insert(0, str(ROOT / "scripts" / "ppo_bc_ablation"))
import notify as notifier  # noqa: E402

SINGLE = "single_task_eval.py"
COMB = "task_combination_eval.py"
SWIT = "task_switching_eval.py"

# (config name under scripts/eval/configs/, eval script). Order: single, comb, swit.
EVALS: list[tuple[str, str]] = [
    ("bc_ablation_single_task_best", SINGLE),
    ("bc_ablation_single_task_final", SINGLE),
    ("bc_ablation_combination_best", COMB),
    ("bc_ablation_combination_final", COMB),
    ("bc_ablation_switching_best", SWIT),
    ("bc_ablation_switching_final", SWIT),
]


def fmt_dur(seconds: float) -> str:
    """Compact human duration, e.g. 2h13m / 7m04s / 42s."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def load_notify_cfg() -> dict:
    """The ``notify`` block from the ablation sweep.yaml (channel + webhook +
    mention id). Returns a 'none' channel if the file/block is absent, which makes
    the notifier fall back to a local banner."""
    if SWEEP_YAML.exists():
        with open(SWEEP_YAML) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("notify") or {"channel": "none"}
    return {"channel": "none"}


def run_with_heartbeat(cmd: list[str], env: dict, on_heartbeat, interval_s: float) -> int:
    """Run ``cmd`` with stdout/stderr inherited (so the eval's live progress shows
    in this terminal), firing ``on_heartbeat()`` every ``interval_s`` seconds until
    it exits. Returns the process exit code."""
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    while True:
        try:
            return proc.wait(timeout=interval_s)
        except subprocess.TimeoutExpired:
            on_heartbeat()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the 6 BC-ablation evals back to back with Discord notifications."
    )
    ap.add_argument(
        "--heartbeat-s",
        type=float,
        default=3600.0,
        help="seconds between heartbeat notifications while an eval runs (default 3600 = 1h)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and exit without running anything",
    )
    args = ap.parse_args()

    notify_cfg = load_notify_cfg()
    env = os.environ.copy()
    # src/ is the import root for utils/wrappers/mdp; prepend it so the eval scripts
    # (and their spawn-started workers, which inherit env) resolve imports.
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [str(ROOT / "src"), env.get("PYTHONPATH", "")] if p
    )

    n = len(EVALS)
    print(f"BC-ablation eval runner: {n} configs, heartbeat every {fmt_dur(args.heartbeat_s)}")
    for i, (cfg_name, script) in enumerate(EVALS, 1):
        print(f"  [{i}/{n}] {script:<24} {cfg_name}")
    print(f"notify channel: {notify_cfg.get('channel', 'none')}")
    if args.dry_run:
        return 0

    overall_start = time.monotonic()
    results: list[tuple[str, int, float]] = []  # (cfg_name, returncode, duration_s)

    notifier.notify(
        "🚀 BC-ablation evals started",
        f"Running {n} configs back to back (single → comb → swit). "
        f"Heartbeat every {fmt_dur(args.heartbeat_s)}; you'll be pinged when all finish.",
        notify_cfg,
    )

    for i, (cfg_name, script) in enumerate(EVALS, 1):
        cmd = [sys.executable, str(EVAL_DIR / script), cfg_name]
        tag = f"[{i}/{n}] {cfg_name}"
        start = time.monotonic()
        print(f"\n=== {tag}: {script} ===", flush=True)

        def heartbeat(_start=start, _tag=tag, _i=i):
            failed = sum(1 for _, rc, _ in results if rc != 0)
            notifier.notify(
                "⏳ BC-ablation eval running",
                f"{_tag} — elapsed {fmt_dur(time.monotonic() - _start)} "
                f"(overall {fmt_dur(time.monotonic() - overall_start)}). "
                f"{_i - 1}/{n} done so far"
                + (f", {failed} failed" if failed else "")
                + ".",
                notify_cfg,
            )

        try:
            rc = run_with_heartbeat(cmd, env, heartbeat, args.heartbeat_s)
        except KeyboardInterrupt:
            notifier.notify(
                "🛑 BC-ablation evals interrupted",
                f"Stopped during {tag} (Ctrl-C). {len(results)}/{n} finished before stopping.",
                notify_cfg,
                mention=True,
            )
            return 130

        dur = time.monotonic() - start
        results.append((cfg_name, rc, dur))

        if rc == 0:
            print(f"=== {tag}: done in {fmt_dur(dur)} ===", flush=True)
            notifier.notify(
                "✅ BC-ablation eval finished",
                f"{tag} done in {fmt_dur(dur)}. ({i}/{n} complete)",
                notify_cfg,
            )
        else:
            print(f"=== {tag}: FAILED (exit {rc}) after {fmt_dur(dur)} ===", flush=True)
            notifier.notify(
                "❌ BC-ablation eval FAILED",
                f"{tag} exited with code {rc} after {fmt_dur(dur)}. "
                f"Continuing with the remaining configs.",
                notify_cfg,
                mention=True,
            )

    total = time.monotonic() - overall_start
    passed = [c for c, rc, _ in results if rc == 0]
    failed = [c for c, rc, _ in results if rc != 0]
    summary = [f"Total time {fmt_dur(total)}.", f"✅ {len(passed)}/{n} passed."]
    if failed:
        summary.append("❌ failed: " + ", ".join(failed))
    notifier.notify(
        "🏁 All BC-ablation evals finished",
        "\n".join(summary),
        notify_cfg,
        mention=True,
    )
    print("\n" + "\n".join(summary))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
