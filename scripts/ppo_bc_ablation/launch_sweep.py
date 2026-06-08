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
budget. As routines finish they free their RAM and the next is admitted.

Because routines also GROW after admission (pretrain's DAgger buffer is unbounded),
admission alone can't stop a mid-flight OOM. A reactive safety net handles that:
when free RAM drops below ``pause.pause_avail_gb`` the governor SIGSTOPs the newest
routines (their whole worker tree) so the kernel swaps their cold pages out, and
SIGCONTs them once RAM recovers above ``pause.resume_avail_gb`` (hysteresis +
min-dwell to avoid flapping). At least one routine always stays running. Monitoring
is cheap: a per-tick ``virtual_memory()`` syscall, plus a per-routine RSS tree-walk
only every ``sample_interval_s``.

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
import threading
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
               "paused": "blue", "done": "green", "failed": "bold red"}


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

    counts = {"queued": 0, "running": 0, "paused": 0, "done": 0, "failed": 0}
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
        f"running {counts['running']}  paused {counts['paused']}  "
        f"done {counts['done']}  failed {counts['failed']}  queued {counts['queued']}   "
        f"elapsed {elapsed}",
        style="cyan",
    )
    if gov:
        gline = Text(
            f"governor  budget {gov['budget'] / 1e9:.0f} GB   "
            f"avail {gov['available'] / 1e9:.0f} GB   "
            f"admitted {gov['admitted']}  paused {gov.get('paused', 0)}   "
            f"est/routine {gov['est'] / 1e9:.1f} GB"
            + ("   [admission held]" if gov.get("held") else "")
            + ("   [PAUSING — low RAM]" if gov.get("pressure") else ""),
            style="magenta",
        )
        return Group(table, footer, gline)
    return Group(table, footer)


# ----------------------------------------------------------- notifications

# Stage emoji for the completion feed / heartbeat table header.
STAGE_EMOJI = {"pretrain": "🧠", "critic": "📉", "rl": "🤖"}
_DISCORD_LIMIT = 1800  # leave headroom under Discord's 2000-char content cap


def notify_async(title: str, message: str, cfg: dict | None,
                 mention: bool = False) -> None:
    """Fire a notification from a daemon thread so the blocking urlopen (15s
    timeout) never stalls the 4 Hz dashboard loop. Best-effort, never raises."""
    threading.Thread(target=notifier.notify, args=(title, message, cfg),
                     kwargs={"mention": mention}, daemon=True).start()


def _chunk_lines(lines: list[str], limit: int = _DISCORD_LIMIT) -> list[list[str]]:
    """Split lines into groups whose joined length stays under ``limit``."""
    chunks, cur, cur_len = [], [], 0
    for ln in lines:
        if cur and cur_len + len(ln) + 1 > limit:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(ln)
        cur_len += len(ln) + 1
    if cur:
        chunks.append(cur)
    return chunks or [[]]


def collect_terminal_stages(work: list[dict]) -> set[tuple[str, str, str]]:
    """Set of (model_id, stage, state) for every stage in a terminal state
    (done | failed), read from the per-routine status files."""
    out: set[tuple[str, str, str]] = set()
    for w in work:
        stages = (read_status(w["status_file"]) or {}).get("stages", {})
        for s in STAGES:
            state = stages.get(s, {}).get("state")
            if state in ("done", "failed"):
                out.add((w["model_id"], s, state))
    return out


def send_stage_completions(new_events: set[tuple[str, str, str]],
                           label_of: dict[str, str], cfg: dict | None,
                           sync: bool = False) -> None:
    """Batch message for stages that reached a terminal state this interval.
    ``sync=True`` blocks (used for the final flush, so the last completions are
    delivered before the process exits and daemon threads are torn down)."""
    send = notifier.notify if sync else notify_async
    # order: by model label, then pipeline stage order
    order = {s: i for i, s in enumerate(STAGES)}
    rows = sorted(new_events, key=lambda e: (label_of.get(e[0], e[0]), order[e[1]]))
    # build (line, important) — only failures and whole-routine completions (rl
    # done) are "important" enough to @-ping; intermediate stages post silently.
    entries: list[tuple[str, bool]] = []
    for model_id, stage, state in rows:
        label = label_of.get(model_id, model_id)
        if state == "failed":
            entries.append((f"❌ `{label}` failed at **{stage}**", True))
        elif stage == "rl":  # last stage -> whole routine done
            entries.append((f"🎉 `{label}` finished **{stage}** — routine complete!", True))
        else:
            entries.append((f"{STAGE_EMOJI.get(stage, '✅')} `{label}` finished **{stage}**", False))
    # chunk under the Discord cap, pinging a chunk only if it carries an important event
    chunks: list[list[tuple[str, bool]]] = []
    cur, cur_len = [], 0
    for line, imp in entries:
        if cur and cur_len + len(line) + 1 > _DISCORD_LIMIT:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append((line, imp))
        cur_len += len(line) + 1
    if cur:
        chunks.append(cur)
    for i, ch in enumerate(chunks):
        title = "✅ Stage updates" + (f" ({i + 1}/{len(chunks)})" if len(chunks) > 1 else "")
        send(title, "\n".join(line for line, _ in ch), cfg,
             mention=any(imp for _, imp in ch))


def _stage_cell(stg: dict, done: str = "✓", fail: str = "x") -> str:
    """Per-stage cell: a checkmark when done, a cross when failed, otherwise the
    percentage complete (0% = not started). ``done``/``fail`` are configurable so
    a code-block-aligned text variant (e.g. ✓/x) can be swapped in."""
    state = stg.get("state", "pending")
    if state == "done":
        return done
    if state == "failed":
        return fail
    return f"{int(stg.get('frac', 0.0) * 100)}%"


def build_status_table(work: list[dict], done: str = "✓", fail: str = "x") -> list[str]:
    """Monospace rows — model x the three stage cells — for a code-block table."""
    head = f"{'model':<10}{'pre':>5}{'cri':>5}{'rl':>5}"
    rows = [head, "-" * len(head)]
    for w in work:
        stages = (read_status(w["status_file"]) or {}).get("stages", {})
        c = [_stage_cell(stages.get(s, {}), done, fail) for s in STAGES]
        rows.append(f"{w['label']:<10}{c[0]:>5}{c[1]:>5}{c[2]:>5}")
    return rows


def server_state_line() -> str:
    """One-line CPU + RAM snapshot for the check-in header."""
    cpu = psutil.cpu_percent(interval=0.3)
    vm = psutil.virtual_memory()
    return (f"🖥️ CPU {cpu:.0f}%  ·  "
            f"RAM {vm.used / 1e9:.0f}/{vm.total / 1e9:.0f} GB ({vm.percent:.0f}%)")


def send_checkin(work: list[dict], start: float, cfg: dict | None,
                 done: str = "✓", fail: str = "x") -> None:
    """Periodic status check-in: timestamp + server state + per-state counts
    (each on its own line) + a tabular snapshot (code block, chunked under
    Discord's content cap)."""
    counts: dict[str, int] = {}
    for w in work:
        counts[w["sched"]] = counts.get(w["sched"], 0) + 1
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    summary = "\n".join([
        f"🕐 {time.strftime('%Y-%m-%d %H:%M:%S')}  ·  ⏱️ {elapsed} elapsed",
        server_state_line(),
        "",
        f"▶️ running {counts.get('running', 0)}",
        f"⏸️ paused {counts.get('paused', 0)}",
        f"✅ done {counts.get('done', 0)}",
        f"❌ failed {counts.get('failed', 0)}",
        f"⏳ queued {counts.get('queued', 0)}",
    ])
    rows = build_status_table(work, done, fail)
    # reserve room for the summary + code fences on the first chunk
    parts = _chunk_lines(rows, _DISCORD_LIMIT - len(summary) - 16)
    for i, ch in enumerate(parts):
        title = "📋 Sweep check-in" + (f" ({i + 1}/{len(parts)})" if len(parts) > 1 else "")
        body = (summary + "\n" if i == 0 else "") + "```\n" + "\n".join(ch) + "\n```"
        notify_async(title, body, cfg)


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
    stage_updates = notify_cfg.get("stage_updates", True)
    stage_flush_s = float(notify_cfg.get("stage_flush_s", 60))
    checkin_s = float(notify_cfg.get("checkin_s") or notify_cfg.get("heartbeat_s") or 900)
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

    # reactive pause safety net (handles routines that grow after admission)
    pause_cfg = gov_cfg.get("pause") or {}
    pause_enabled = gov_enabled and pause_cfg.get("enabled", True)
    pause_avail = float(pause_cfg.get("pause_avail_gb", 8)) * 1e9
    resume_avail = float(pause_cfg.get("resume_avail_gb", 16)) * 1e9
    min_dwell = float(pause_cfg.get("min_dwell_s", 15))

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
    running: list[tuple[dict, subprocess.Popen, object]] = []   # admission order (oldest first)
    paused: list[tuple[dict, subprocess.Popen, object]] = []     # SIGSTOPped, swapped out
    peak_est: dict[str, float] = {}   # model_id -> rolling-max tree RSS (bytes)
    start = time.time()
    last_sample = 0.0
    last_pause_action = 0.0            # for pause/resume min-dwell (anti-flap)
    cold_start_pending = False         # True between an admit and the next RSS sample
    # notification feed: seed the seen-set with stages already terminal (so a
    # resume doesn't replay old completions), and start both timers now.
    label_of = {w["model_id"]: w["label"] for w in work}
    seen_terminal = collect_terminal_stages(work)
    last_stage_flush = start
    last_checkin = start
    psutil.cpu_percent()  # prime the CPU% meter
    gov_info = {"budget": budget, "available": psutil.virtual_memory().available,
                "admitted": 0, "paused": 0, "est": per_routine_est,
                "held": False, "pressure": False}

    def can_admit(used: float) -> bool:
        if hard_cap and len(running) >= hard_cap:
            return False
        if paused:
            return False  # under pressure: resume shelved routines before adding new
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
            while queue or running or paused:
                vm = psutil.virtual_memory()  # cheap per-tick syscall
                now = time.time()

                # periodic RSS tree-walk (the only expensive sampling); only the
                # un-paused routines — paused ones get swapped out, so their RSS is
                # meaningless; keep their last (running) peak frozen.
                if now - last_sample >= sample_interval:
                    for w, proc, _ in running:
                        rss = routine_tree_rss(proc)
                        peak_est[w["model_id"]] = max(peak_est.get(w["model_id"], 0.0), rss)
                    last_sample = now
                    cold_start_pending = False

                # reactive pause/resume: shelve the newest routine when free RAM is
                # low so the kernel can swap it out; un-shelve when RAM recovers.
                # Hysteresis (pause<resume) + min-dwell keep it from flapping; never
                # pause the last running routine so progress can't stall.
                if pause_enabled and paused and not running:
                    # forced resume: never leave zero running while work is shelved
                    # (free RAM can't recover with nothing running -> would deadlock
                    # in the hysteresis dead-zone). Ignores dwell on purpose.
                    w, proc, fh = paused.pop(0)
                    signal_tree(proc, signal.SIGCONT)
                    w["sched"] = "running"
                    running.insert(0, (w, proc, fh))
                    last_pause_action = now
                elif pause_enabled and (now - last_pause_action) >= min_dwell:
                    if vm.available < pause_avail and len(running) > 1:
                        w, proc, fh = running.pop()        # newest-admitted
                        signal_tree(proc, signal.SIGSTOP)
                        w["sched"] = "paused"
                        paused.append((w, proc, fh))
                        last_pause_action = now
                    elif paused and vm.available > resume_avail:
                        w, proc, fh = paused.pop(0)        # longest-shelved first
                        signal_tree(proc, signal.SIGCONT)
                        w["sched"] = "running"
                        running.insert(0, (w, proc, fh))   # keep admission order
                        last_pause_action = now

                # admission: launch while memory (and any hard cap) allows
                while queue and can_admit(vm.used):
                    w = queue.pop(0)
                    proc, fh = launch(w, timesteps_override)
                    w["sched"] = "running"
                    running.append((w, proc, fh))
                    peak_est[w["model_id"]] = per_routine_est  # seed
                    cold_start_pending = True
                    vm = psutil.virtual_memory()  # refresh before the next admit decision

                # reap finished (only un-paused routines can exit; poll paused too
                # in case one was killed externally while stopped)
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
                still_paused = []
                for w, proc, fh in paused:
                    rc = proc.poll()
                    if rc is None:
                        still_paused.append((w, proc, fh))
                    else:
                        fh.close()
                        w["sched"] = "done" if rc == 0 else "failed"
                        peak_est.pop(w["model_id"], None)
                paused = still_paused

                gov_info.update(available=vm.available, admitted=len(running),
                                paused=len(paused),
                                est=max(per_routine_est, max(peak_est.values(), default=0.0)),
                                held=bool(queue) and not can_admit(vm.used),
                                pressure=bool(paused) or vm.available < pause_avail)
                live.update(render(work, start, gov_info))

                # batched stage-completion feed (~every stage_flush_s)
                if stage_updates and (now - last_stage_flush) >= stage_flush_s:
                    current = collect_terminal_stages(work)
                    new_events = current - seen_terminal
                    if new_events:
                        send_stage_completions(new_events, label_of, notify_cfg)
                        seen_terminal = current
                    last_stage_flush = now
                # periodic tabular check-in (~every checkin_s)
                if (now - last_checkin) >= checkin_s:
                    send_checkin(work, start, notify_cfg)
                    last_checkin = now

                time.sleep(0.25)
            # final flush so the last stage completions aren't lost to the timer
            if stage_updates:
                new_events = collect_terminal_stages(work) - seen_terminal
                if new_events:
                    send_stage_completions(new_events, label_of, notify_cfg, sync=True)
            live.update(render(work, start, gov_info))
    except KeyboardInterrupt:
        console.print("\n[red]interrupted — terminating routines (and their worker trees)...[/red]")
        # SIGCONT paused trees first: a stopped process won't act on SIGTERM until continued.
        for w, proc, fh in running + paused:
            signal_tree(proc, signal.SIGCONT)
            signal_tree(proc, signal.SIGTERM)
        deadline = time.time() + 5.0
        for w, proc, fh in running + paused:
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
    title = "🏁 BC_Coef ablation sweep finished"
    msg = (f"{len(done)}/{len(work)} routines succeeded in {elapsed}." +
           (f"\nFailed: {', '.join(w['model_id'] for w in failed)}" if failed else ""))
    console.print(f"\n[bold]{title}[/bold]\n{msg}")
    notifier.notify(title, msg, notify_cfg, mention=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
