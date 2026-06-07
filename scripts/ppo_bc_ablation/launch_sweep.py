"""
scripts/ppo_bc_ablation/launch_sweep.py  (launch script 2)
==========================================================

Fan ``launch_routine.py`` out across the BC_Coef ablation sweep and supervise it
with a live terminal dashboard. By default this is 15 coefficients (0.1..1.5)
times the two lineages (uniform + adversarial) = 30 routines, each a 3-stage
pretrain -> critic -> rl pipeline.

Memory governor (admission control)
-----------------------------------
Each routine's stages run with the BLAS thread pools capped to 1
(``OMP_NUM_THREADS=1`` etc., set in train.py / launch_routine.py), so the way to
peg every core is to oversubscribe at the *process* level. The only real ceiling
is RAM: each active routine forks ~15-20 SubprocVecEnv workers, so launching all
30 at once OOM-kills the box. Instead of a fixed ``max_parallel``, the governor
admits routines greedily up to a RAM budget of ``total - headroom_gb`` (default
16 GB): a new routine launches only while ``used + projected_peak`` fits the
budget. Running routines are never paused — they finish (fully freeing their RAM)
and that headroom admits the next one. Monitoring is cheap: a per-tick
``virtual_memory()`` syscall, plus a per-routine RSS tree-walk only every
``sample_interval_s``.

Resume
------
With a stable ``run_tag`` (default ``"resumable"``) the run dir persists, so a
relaunch skips routines/stages already finished (verified by a valid ``final.zip``
on disk). Model artifacts under ``models/`` are namespace-keyed and survive
regardless. Pass ``--no-resume`` to ignore prior progress.

Everything is configured from ``sweep.yaml`` (no environment variables); see
``sweep.example.yaml`` for the schema and the notification setup.

    python scripts/ppo_bc_ablation/launch_sweep.py
    python scripts/ppo_bc_ablation/launch_sweep.py --config my_sweep.yaml --no-resume
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import psutil
import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

# src/ is editable-installed; utils.paths is pure pathlib (no torch).
from utils.paths import SAVE_DIR, MODELS_DIR, DATASET_DIR

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


# ------------------------------------------------------------------- resume


def _valid_model_zip(path: Path) -> bool:
    """A finished stage leaves a complete SB3 zip containing ``policy.pth``; a
    crash mid-save (not atomic) can leave a truncated one. Guard against trusting
    a partial artifact on resume."""
    try:
        if not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as zf:
            return "policy.pth" in zf.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def routine_complete(model_id: str) -> bool:
    """True iff all three stages' artifacts already exist on disk (so the whole
    routine can be shown done without relaunching it). Mirrors
    ``launch_routine.stage_is_complete`` for the three stages."""
    for stage in STAGES:
        if not _valid_model_zip(MODELS_DIR / model_id / stage / "final.zip"):
            return False
    return (DATASET_DIR / f"{model_id}/pretrain.npz").exists()


# ----------------------------------------------------------------- governor


def routine_tree_rss(proc: subprocess.Popen) -> int:
    """Sum RSS over the routine's main process and its whole SubprocVecEnv worker
    tree. Each access is guarded — children die between enumeration and read."""
    try:
        p = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return 0
    procs = [p]
    try:
        procs += p.children(recursive=True)
    except psutil.NoSuchProcess:
        pass
    total = 0
    for q in procs:
        try:
            total += q.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def signal_tree(proc: subprocess.Popen, sig: int) -> None:
    """Deliver ``sig`` to the routine's whole process group (each routine is a
    session leader via ``start_new_session=True``), so the ~15-20 worker
    processes are signalled too — not just the routine main. Falls back to a
    psutil tree-walk if the group is already gone."""
    try:
        os.killpg(os.getpgid(proc.pid), sig)
        return
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        p = psutil.Process(proc.pid)
        targets = p.children(recursive=True) + [p]
    except psutil.NoSuchProcess:
        return
    for q in targets:
        try:
            q.send_signal(sig)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


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


def render(work: list[dict], start: float, gov: dict | None = None) -> Group:
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
    if gov:
        gline = Text(
            f"governor  budget {gov['budget'] / 1e9:.0f} GB   "
            f"avail {gov['available'] / 1e9:.0f} GB   "
            f"admitted {gov['admitted']}   "
            f"est/routine {gov['est'] / 1e9:.1f} GB"
            + ("   [admission held]" if gov.get("held") else ""),
            style="magenta",
        )
        return Group(table, footer, gline)
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
    # start_new_session=True makes each routine a process-group leader so the
    # governor can signal its whole SubprocVecEnv worker tree (see signal_tree).
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    return proc, fh


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the BC_Coef ablation sweep with a live dashboard.")
    ap.add_argument("--config", default=None, help="sweep config YAML (default: sweep.yaml then sweep.example.yaml)")
    ap.add_argument("--max-parallel", type=int, default=None, help="hard cap on concurrent routines (on top of the memory budget)")
    ap.add_argument("--run-tag", default=None, help="override config run_tag (names the run dir)")
    ap.add_argument("--no-resume", action="store_true", help="ignore prior progress; do not skip finished routines")
    args = ap.parse_args()

    console = Console()
    cfg, cfg_path = load_config(args.config)
    notify_cfg = cfg.get("notify") or {"channel": "none"}
    timesteps_override = cfg.get("timesteps_override")

    # --- governor config (admission control) ---
    gov_cfg = cfg.get("governor") or {}
    gov_enabled = gov_cfg.get("enabled", True)
    budget = psutil.virtual_memory().total - float(gov_cfg.get("headroom_gb", 16)) * 1e9
    per_routine_est = float(gov_cfg.get("per_routine_est_gb", 6)) * 1e9
    sample_interval = float(gov_cfg.get("sample_interval_s", 3))
    # optional hard ceiling: CLI > top-level max_parallel > governor.max_parallel
    hard_cap = args.max_parallel or cfg.get("max_parallel") or gov_cfg.get("max_parallel")
    hard_cap = int(hard_cap) if hard_cap else None

    # --- resume config ---
    resume_cfg = cfg.get("resume") or {}
    resume_enabled = resume_cfg.get("enabled", True) and not args.no_resume
    run_tag = (args.run_tag or resume_cfg.get("run_tag")
               or cfg.get("run_tag") or time.strftime("%Y%m%d_%H%M%S"))
    run_dir = SAVE_DIR / "bc_ablation_sweep" / run_tag
    (run_dir / "status").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    work = build_worklist(cfg, run_dir)

    # resume: mark already-finished routines done so they're never relaunched.
    # (Partially-finished routines are queued; launch_routine skips their done
    # stages internally.) Trust on-disk artifacts, not the status flag.
    skipped = 0
    if resume_enabled:
        for w in work:
            if routine_complete(w["model_id"]):
                w["sched"] = "done"
                skipped += 1

    with open(run_dir / "meta.json", "w") as f:
        json.dump({"config_path": str(cfg_path), "run_tag": run_tag,
                   "n_routines": len(work), "hard_cap": hard_cap,
                   "budget_gb": round(budget / 1e9, 1),
                   "resume_enabled": resume_enabled, "resumed_done": skipped,
                   "timesteps_override": timesteps_override,
                   "models": [w["model_id"] for w in work]}, f, indent=2)

    console.print(f"[bold]BC_Coef ablation sweep[/bold]  config={cfg_path.name}  "
                  f"routines={len(work)}  budget={budget / 1e9:.0f} GB"
                  + (f"  hard_cap={hard_cap}" if hard_cap else "")
                  + (f"  governor={'on' if gov_enabled else 'off'}"))
    if resume_enabled and skipped:
        console.print(f"[green]resume: {skipped}/{len(work)} routines already complete — skipping[/green]")
    console.print(f"run dir: {run_dir}")
    console.print(f"per-model logs: {run_dir / 'logs'}\n")

    queue = [w for w in work if w["sched"] == "queued"]
    running: list[tuple[dict, subprocess.Popen, object]] = []
    peak_est: dict[str, float] = {}   # model_id -> rolling-max tree RSS (bytes)
    start = time.time()
    last_sample = 0.0
    cold_start_pending = False         # True between an admit and the next RSS sample
    psutil.cpu_percent()  # prime the CPU% meter
    gov_info = {"budget": budget, "available": psutil.virtual_memory().available,
                "admitted": 0, "est": per_routine_est, "held": False}

    def can_admit(used: float) -> bool:
        if hard_cap and len(running) >= hard_cap:
            return False
        if not running:
            return True  # floor: always keep at least one routine moving
        if not gov_enabled:
            return True  # governor off -> only the hard_cap gates admission
        if cold_start_pending:
            return False  # let one cold-start's RAM ramp register before the next
        projected = max(per_routine_est, max(peak_est.values(), default=0.0))
        return used + projected <= budget

    try:
        with Live(render(work, start, gov_info), console=console,
                  refresh_per_second=4, screen=False) as live:
            while queue or running:
                vm = psutil.virtual_memory()  # cheap per-tick syscall

                # periodic RSS tree-walk (the only expensive sampling)
                now = time.time()
                if now - last_sample >= sample_interval:
                    for w, proc, _ in running:
                        rss = routine_tree_rss(proc)
                        peak_est[w["model_id"]] = max(peak_est.get(w["model_id"], 0.0), rss)
                    last_sample = now
                    cold_start_pending = False

                # admission: launch while memory (and any hard cap) allows
                while queue and can_admit(vm.used):
                    w = queue.pop(0)
                    proc, fh = launch(w, timesteps_override)
                    w["sched"] = "running"
                    running.append((w, proc, fh))
                    peak_est[w["model_id"]] = per_routine_est  # seed
                    cold_start_pending = True
                    vm = psutil.virtual_memory()  # refresh before the next admit decision

                # reap finished
                still = []
                for w, proc, fh in running:
                    rc = proc.poll()
                    if rc is None:
                        still.append((w, proc, fh))
                    else:
                        fh.close()
                        w["sched"] = "done" if rc == 0 else "failed"
                        peak_est.pop(w["model_id"], None)
                        cold_start_pending = False  # a completion frees RAM now
                running = still

                gov_info.update(available=vm.available, admitted=len(running),
                                est=max(per_routine_est, max(peak_est.values(), default=0.0)),
                                held=bool(queue) and not can_admit(vm.used))
                live.update(render(work, start, gov_info))
                time.sleep(0.25)
            live.update(render(work, start, gov_info))
    except KeyboardInterrupt:
        console.print("\n[red]interrupted — terminating running routines (and their worker trees)...[/red]")
        for w, proc, fh in running:
            signal_tree(proc, signal.SIGTERM)
        deadline = time.time() + 5.0
        for w, proc, fh in running:
            try:
                proc.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                signal_tree(proc, signal.SIGKILL)
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
