"""
scripts.ppo_bc_ablation.progress
=================================

Cross-process progress reporting for the BC_Coef ablation sweep.

Each ``launch_routine.py`` process owns one status JSON file describing where its
3-stage routine (pretrain -> critic -> rl) is. ``launch_sweep.py`` only ever
*reads* those files to render the dashboard — so writes are single-writer and we
just need them to be atomic (temp file + ``os.replace``) so a reader never sees a
half-written file.

Status schema::

    {
      "model_id": "bc_ablation/bc_0.10",
      "bc_coef": 0.1,
      "adversarial": false,
      "overall_state": "queued|running|done|failed",
      "error": null,
      "updated_at": 1234567890.0,
      "stages": {
        "pretrain": {"state": "pending|running|done|failed", "frac": 0.0,
                     "num_timesteps": 0, "total": 4300800},
        "critic":   {...},
        "rl":       {...}
      }
    }
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

STAGES = ("pretrain", "critic", "rl")


# --------------------------------------------------------------------------- IO


def _atomic_write(path: str | os.PathLike, data: dict) -> None:
    """Write ``data`` as JSON to ``path`` atomically (temp file in the same dir +
    ``os.replace``), so a concurrent reader always sees a complete document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_status(path: str | os.PathLike) -> dict | None:
    """Best-effort read; returns None if the file is missing or mid-write garbage."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def new_status(path, model_id: str, bc_coef: float, adversarial: bool,
               totals: dict[str, int]) -> dict:
    """Initialise a status file with all three stages pending and write it."""
    status = {
        "model_id": model_id,
        "bc_coef": bc_coef,
        "adversarial": adversarial,
        "overall_state": "queued",
        "error": None,
        "updated_at": time.time(),
        "stages": {
            s: {"state": "pending", "frac": 0.0, "num_timesteps": 0,
                "total": int(totals.get(s, 0))}
            for s in STAGES
        },
    }
    _atomic_write(path, status)
    return status


def update_overall(path, *, state: str | None = None, error: str | None = None) -> None:
    status = read_status(path) or {}
    if state is not None:
        status["overall_state"] = state
    if error is not None:
        status["error"] = error
    status["updated_at"] = time.time()
    _atomic_write(path, status)


def set_stage(path, stage_key: str, *, state: str | None = None,
              frac: float | None = None, num_timesteps: int | None = None,
              total: int | None = None) -> None:
    status = read_status(path)
    if status is None:
        return
    st = status["stages"].setdefault(stage_key, {"state": "pending", "frac": 0.0})
    if state is not None:
        st["state"] = state
        # snap the bar to the edges on entry/exit so a stage reads 0% pending / 100% done
        if state == "done":
            st["frac"] = 1.0
        elif state == "running" and st.get("frac", 0.0) == 0.0:
            st["frac"] = 0.0
    if frac is not None:
        st["frac"] = max(0.0, min(1.0, frac))
    if num_timesteps is not None:
        st["num_timesteps"] = int(num_timesteps)
    if total is not None:
        st["total"] = int(total)
    status["updated_at"] = time.time()
    _atomic_write(path, status)


# ---------------------------------------------------------------------- callback


class ProgressWriter(BaseCallback):
    """Writes the active stage's progress fraction to a status file each rollout.

    Cadence is one write per rollout (n_steps x n_envs steps) — cheap, and avoids
    parsing tqdm. ``total`` is the stage's ``cfg.timesteps`` (the model is freshly
    constructed per stage so ``num_timesteps`` runs 0 -> total).
    """

    def __init__(self, status_file: str, stage_key: str, total_timesteps: int):
        super().__init__()
        self.status_file = status_file
        self.stage_key = stage_key
        self.total = max(1, int(total_timesteps))

    def _write(self) -> None:
        set_stage(
            self.status_file,
            self.stage_key,
            frac=self.num_timesteps / self.total,
            num_timesteps=self.num_timesteps,
            total=self.total,
        )

    def _on_training_start(self) -> None:
        set_stage(self.status_file, self.stage_key, state="running",
                  num_timesteps=0, total=self.total)

    def _on_rollout_end(self) -> None:
        self._write()

    def _on_step(self) -> bool:  # required abstract method
        return True
