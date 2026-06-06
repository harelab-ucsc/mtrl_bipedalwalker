"""
scripts/ppo_bc_ablation/launch_sweep.py  (launch script 2)
==========================================================

Fan ``launch_routine.py`` out across the BC_Coef ablation sweep and supervise it
with a live terminal dashboard. By default this is 15 coefficients (0.1..1.5)
times the two lineages (uniform + adversarial) = 30 routines, each a 3-stage
pretrain -> critic -> rl pipeline.

CPU saturation
--------------
Each routine's stages run with the BLAS thread pools capped to 1
(``OMP_NUM_THREADS=1`` etc., set in train.py / launch_routine.py), so the way to
peg every core is to oversubscribe at the *process* level: by default ALL
routines launch at once. On a Threadripper-class box (24c/48t) that keeps every
core near 100% (verify in htop). The only real ceiling is RAM (~1-2 GB per active
routine's main process) — set ``max_parallel`` in the config (or --max-parallel)
to throttle if memory-bound.

Everything is configured from ``sweep.yaml`` (no environment variables); see
``sweep.example.yaml`` for the schema and the notification setup.

    python scripts/ppo_bc_ablation/launch_sweep.py
    python scripts/ppo_bc_ablation/launch_sweep.py --config my_sweep.yaml --max-parallel 6
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

# src/ is editable-installed; utils.paths is pure pathlib (no torch).
from utils.paths import SAVE_DIR

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import notify as notifier  # stdlib-only module

LAUNCH_ROUTINE = _HERE / "launch_routine.py"
STAGES = ("pretrain", "critic", "rl")
STATE_STYLE = {"queued": "grey50", "pending": "grey50", "running": "yellow",
               "done": "green", "failed": "bold red"}


# ------------------------------------------------------------------- config


def load_config(path_arg: str | None) -> tuple[dict, Path]:
    """Load sweep.yaml, falling back to sweep.example.yaml. Returns (cfg, path)."""
    if path_arg:
        p = Path(path_arg)
    else:
        p = _HERE / "sweep.yaml"
        if not p.exists():
            p = _HERE / "sweep.example.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no sweep config found ({p})")
    with open(p) as f:
        return yaml.safe_load(f), p


def coef_str(coef: float) -> str:
    return f"{coef:.2f}"


def namespace(coef: float, adversarial: bool) -> str:
    return f"bc_ablation{'_adv' if adversarial else ''}/bc_{coef_str(coef)}"


def build_worklist(cfg: dict, run_dir: Path) -> list[dict]:
    coefs = cfg.get("bc_coefs") or [round(0.1 * i, 2) for i in range(1, 16)]
    lineages = cfg.get("lineages") or ["uniform", "adversarial"]
    adv_for = {"uniform": False, "adversarial": True}
    work = []
    for lin in lineages:
        adv = adv_for[lin]
        for coef in coefs:
            ns = namespace(coef, adv)
            safe = ns.replace("/", "__")
            work.append({
                "coef": float(coef),
                "adversarial": adv,
                "model_id": ns,
                "label": f"{coef_str(coef)} {'adv' if adv else 'uni'}",
                "status_file": str(run_dir / "status" / f"{safe}.json"),
                "log_file": str(run_dir / "logs" / f"{safe}.log"),
                "sched": "queued",   # queued | running | done | failed
            })
    return work


def read_status(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------- dashboard


def bar(frac: float, state: str, width: int = 16) -> Text:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    style = STATE_STYLE.get(state, "grey50")
    t = Text()
    t.append("█" * filled, style=style)
    t.append("░" * (width - filled), style="grey30")
    t.append(f" {int(frac * 100):3d}%", style=style)
    return t


def render(work: list[dict], start: float) -> Group:
    table = Table(expand=False, pad_edge=False)
    table.add_column("model", no_wrap=True, width=9)
    for s in STAGES:
        table.add_column(s, no_wrap=True, width=21)
    table.add_column("state", no_wrap=True, width=7)

    counts = {"queued": 0, "running": 0, "done": 0, "failed": 0}
    for w in work:
        counts[w["sched"]] = counts.get(w["sched"], 0) + 1
        st = read_status(w["status_file"]) or {}
        stages = st.get("stages", {})
        cells = [Text(w["label"], no_wrap=True)]
        for s in STAGES:
            stg = stages.get(s, {})
            state = stg.get("state", "pending")
            # if the whole routine failed, the unfinished active stage shows red
            if w["sched"] == "failed" and state == "running":
                state = "failed"
            cells.append(bar(float(stg.get("frac", 0.0)), state))
        cells.append(Text(w["sched"], style=STATE_STYLE.get(w["sched"], "white")))
        table.add_row(*cells)

    vm = psutil.virtual_memory()
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    footer = Text(
        f"CPU {psutil.cpu_percent():5.1f}%   "
        f"RAM {vm.percent:4.1f}% ({vm.used / 1e9:.0f}/{vm.total / 1e9:.0f} GB)   "
        f"running {counts['running']}  done {counts['done']}  "
        f"failed {counts['failed']}  queued {counts['queued']}   "
        f"elapsed {elapsed}",
        style="cyan",
    )
    return Group(table, footer)


# --------------------------------------------------------------- scheduling


def launch(w: dict, timesteps_override: int | None) -> tuple[subprocess.Popen, object]:
    Path(w["log_file"]).parent.mkdir(parents=True, exist_ok=True)
    Path(w["status_file"]).parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(LAUNCH_ROUTINE),
           "--bc-coef", repr(w["coef"]),
           "--status-file", w["status_file"]]
    if w["adversarial"]:
        cmd.append("--adversarial")
    if timesteps_override:
        cmd += ["--timesteps-override", str(timesteps_override)]
    fh = open(w["log_file"], "w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return proc, fh


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the BC_Coef ablation sweep with a live dashboard.")
    ap.add_argument("--config", default=None, help="sweep config YAML (default: sweep.yaml then sweep.example.yaml)")
    ap.add_argument("--max-parallel", type=int, default=None, help="override config max_parallel")
    ap.add_argument("--run-tag", default=None, help="override config run_tag (names the run dir)")
    args = ap.parse_args()

    console = Console()
    cfg, cfg_path = load_config(args.config)
    notify_cfg = cfg.get("notify") or {"channel": "none"}
    timesteps_override = cfg.get("timesteps_override")

    run_tag = args.run_tag or cfg.get("run_tag") or time.strftime("%Y%m%d_%H%M%S")
    run_dir = SAVE_DIR / "bc_ablation_sweep" / run_tag
    (run_dir / "status").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    work = build_worklist(cfg, run_dir)
    max_parallel = args.max_parallel or cfg.get("max_parallel") or len(work)
    max_parallel = max(1, min(int(max_parallel), len(work)))

    with open(run_dir / "meta.json", "w") as f:
        json.dump({"config_path": str(cfg_path), "run_tag": run_tag,
                   "n_routines": len(work), "max_parallel": max_parallel,
                   "timesteps_override": timesteps_override,
                   "models": [w["model_id"] for w in work]}, f, indent=2)

    console.print(f"[bold]BC_Coef ablation sweep[/bold]  config={cfg_path.name}  "
                  f"routines={len(work)}  max_parallel={max_parallel}")
    console.print(f"run dir: {run_dir}")
    console.print(f"per-model logs: {run_dir / 'logs'}\n")

    queue = list(work)
    running: list[tuple[dict, subprocess.Popen, object]] = []
    start = time.time()
    psutil.cpu_percent()  # prime the CPU% meter

    try:
        with Live(render(work, start), console=console, refresh_per_second=4, screen=False) as live:
            while queue or running:
                # fill the pool
                while queue and len(running) < max_parallel:
                    w = queue.pop(0)
                    proc, fh = launch(w, timesteps_override)
                    w["sched"] = "running"
                    running.append((w, proc, fh))
                # reap finished
                still = []
                for w, proc, fh in running:
                    rc = proc.poll()
                    if rc is None:
                        still.append((w, proc, fh))
                    else:
                        fh.close()
                        w["sched"] = "done" if rc == 0 else "failed"
                running = still
                live.update(render(work, start))
                time.sleep(0.25)
            live.update(render(work, start))
    except KeyboardInterrupt:
        console.print("\n[red]interrupted — terminating running routines...[/red]")
        for w, proc, fh in running:
            proc.terminate()
            fh.close()
        return 130

    # summary + notification
    done = [w for w in work if w["sched"] == "done"]
    failed = [w for w in work if w["sched"] == "failed"]
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    title = "BC_Coef ablation sweep finished"
    msg = (f"{len(done)}/{len(work)} routines succeeded in {elapsed}." +
           (f"\nFailed: {', '.join(w['model_id'] for w in failed)}" if failed else ""))
    console.print(f"\n[bold]{title}[/bold]\n{msg}")
    notifier.notify(title, msg, notify_cfg)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
