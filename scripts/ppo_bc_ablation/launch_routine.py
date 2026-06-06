"""
scripts/ppo_bc_ablation/launch_routine.py  (launch script 1)
============================================================

Run ONE model's full PPO_BC routine end-to-end for a given BC coefficient and
adversarial setting — the ROS-".launch.py"-style entrypoint for the BC_Coef
ablation. Three stages run back-to-back in this process:

  1. pretrain      single gait tasks only (expert-pollable), BC @ --bc-coef.
                   Identical to PRETRAIN_200 except bc_coef + the adversarial flag.
  2. critic        re-fit a fresh critic on the singles+combos union, fast
                   switching. Identical to pretrain_critic CS_200(A).
  3. rl            PPO_BC RL finetune on the singles+combos union, fast switching,
                   BC pinned @ 0.1. Identical to CS_200A (adv) / CS_200 (uniform).

Adversarial task selection is a single global flag (--adversarial): when set it
is used in BOTH the pretrain and RL stages, and every artifact lands under the
``bc_ablation_adv/`` namespace (vs ``bc_ablation/``) — which is how this repo
"marks" a package as adversarial (a top-level dir convention, not zip metadata).
The critic stage has no adversarial machinery but inherits the namespace, so its
package is marked consistently.

Examples
--------
    # one uniform routine at bc_coef=0.7
    python scripts/ppo_bc_ablation/launch_routine.py --bc-coef 0.7

    # one adversarial routine, just print + assert the 3 configs (no training)
    python scripts/ppo_bc_ablation/launch_routine.py --bc-coef 0.7 --adversarial --dry-run
"""

from __future__ import annotations

# Cap per-process BLAS thread pools BEFORE importing numpy/torch (mirrors train.py).
# Must run before train/pretrain_critic are imported so their own setdefault is a no-op.
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import math
import sys
import traceback
from pathlib import Path

import numpy as np

# --- make the sibling ppo_bc scripts + this dir importable -------------------
_HERE = Path(__file__).resolve().parent
_PPO_BC = _HERE.parent / "ppo_bc"
for _p in (str(_PPO_BC), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# src/ is installed editable, so these resolve globally.
from mdp.bipedal_walker.tasks import SINGLE_TASKS_GAIT, COMBINATION_TASKS_GAIT
from utils.paths import SAVE_DIR

# sibling config presets (scripts/ppo_bc on sys.path)
from train_config import PRETRAIN_200, CS_200, CS_200A
from pretrain_critic_config import CS_200A as CRITIC_CS_200A

import progress
from progress import ProgressWriter

UNION_TASKS = (*SINGLE_TASKS_GAIT, *COMBINATION_TASKS_GAIT)
RL_BC_COEF = 0.1  # pinned in the RL stage, per the ablation design


def coef_str(coef: float) -> str:
    return f"{coef:.2f}"


def namespace(coef: float, adversarial: bool) -> str:
    return f"bc_ablation{'_adv' if adversarial else ''}/bc_{coef_str(coef)}"


def build_stage_configs(coef: float, adversarial: bool, timesteps_override: int | None = None):
    """Derive the three stage configs from the referenced presets via ``__call__``
    overrides. Only experiment_name / warm-start paths / bc_coef / adversarial flag
    change; every env + PPO field is inherited verbatim."""
    ns = namespace(coef, adversarial)

    # Stage 1 — pretrain (PRETRAIN_200): single tasks only, BC carries imitation.
    # PRETRAIN_200 pins adversarial off + eval_steps=0, so turning adv on also needs
    # eval steps so the difficulty PMF can rescore.
    pretrain = PRETRAIN_200(
        experiment_name=f"{ns}/pretrain",
        bc_coef=coef,
        adversarial_ag=adversarial,
        adversarial_eval_steps_per_task=(5000 if adversarial else 0),
    )

    # Stage 2 — critic pretrain (CS_200A): singles+combos union, fast switching.
    # Adversarial is irrelevant here; namespace inheritance does the "marking".
    critic = CRITIC_CS_200A(
        experiment_name=f"{ns}/critic",
        load_actor_from=f"{ns}/pretrain/final.zip",
    )

    # Stage 3 — RL finetune. Pick the base that already carries the right
    # adversarial_ag (CS_200A=True, CS_200=False); both pin bc_coef=0.1.
    rl_base = CS_200A if adversarial else CS_200
    rl = rl_base(
        experiment_name=f"{ns}/rl",
        load_model=f"{ns}/critic/final.zip",
        load_dataset=f"{ns}/pretrain.npz",
    )

    if timesteps_override:
        pretrain = pretrain(timesteps=timesteps_override)
        critic = critic(timesteps=timesteps_override)
        rl = rl(timesteps=timesteps_override)

    return ns, pretrain, critic, rl


def assert_invariants(coef: float, adversarial: bool, ns, pretrain, critic, rl) -> None:
    """Guard the requirements the ablation depends on (used by --dry-run)."""
    log1 = float(np.log(1.0))
    checks = {
        "ns marks adversarial correctly":
            ns.startswith("bc_ablation_adv/") == adversarial,
        "pretrain.bc_coef == coef": math.isclose(pretrain.bc_coef, coef),
        "pretrain is single-tasks-only":
            tuple(pretrain.allowed_task_mixing) == tuple(SINGLE_TASKS_GAIT),
        "pretrain.adversarial_ag == flag": pretrain.adversarial_ag == adversarial,
        "pretrain switching off (>= ep_time)":
            pretrain.task_switching_time >= pretrain.ep_time,
        "critic loads pretrain actor":
            critic.load_actor_from == f"{ns}/pretrain/final.zip",
        "critic trains on the union":
            tuple(critic.allowed_task_mixing) == UNION_TASKS,
        "rl.bc_coef pinned to 0.1": math.isclose(rl.bc_coef, RL_BC_COEF),
        "rl trains on Comb+Switch union": tuple(rl.allowed_task_mixing) == UNION_TASKS,
        "rl fast switching (3s)": math.isclose(rl.task_switching_time, 3.0),
        "rl.adversarial_ag == flag": rl.adversarial_ag == adversarial,
        "rl loads critic + dataset":
            rl.load_model == f"{ns}/critic/final.zip"
            and rl.load_dataset == f"{ns}/pretrain.npz",
        "all stages init_log_std == log(1)":
            math.isclose(pretrain.init_log_std, log1)
            and math.isclose(critic.init_log_std, log1)
            and math.isclose(rl.init_log_std, log1),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError("dry-run invariant(s) failed:\n  - " + "\n  - ".join(failed))


def _print_dry_run(coef, adversarial, ns, pretrain, critic, rl) -> None:
    def line(tag, cfg, extra=""):
        print(f"  {tag:<9} exp={cfg.experiment_name:<32} "
              f"bc_coef={getattr(cfg, 'bc_coef', '-'):<5} adv={getattr(cfg, 'adversarial_ag', '-')!s:<5} "
              f"switch={cfg.task_switching_time}s  log_std={cfg.init_log_std:.3f}  ts={cfg.timesteps:,}{extra}")

    print(f"\n{'=' * 78}")
    print(f"  routine  coef={coef}  adversarial={adversarial}  ->  {ns}")
    print(f"{'=' * 78}")
    line("pretrain", pretrain, f"  tasks={len(pretrain.allowed_task_mixing)} (singles only)")
    line("critic", critic, f"  <- {critic.load_actor_from}  tasks={len(critic.allowed_task_mixing)} (union)")
    line("rl", rl, f"  <- {rl.load_model}  + {rl.load_dataset}  tasks={len(rl.allowed_task_mixing)} (union)")
    print(f"{'=' * 78}\n")


def default_status_file(ns: str) -> str:
    return str(SAVE_DIR / "bc_ablation_runs" / f"{ns.replace('/', '__')}.json")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one full PPO_BC ablation routine (pretrain -> critic -> rl).")
    ap.add_argument("--bc-coef", type=float, required=True, help="pretrain BC coefficient")
    ap.add_argument("--adversarial", action="store_true",
                    help="use adversarial task selection (global: pretrain + rl); else uniform")
    ap.add_argument("--status-file", default=None, help="path for the progress JSON (sweep sets this)")
    ap.add_argument("--timesteps-override", type=int, default=None,
                    help="override per-stage timesteps (smoke tests)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build + print + assert the 3 stage configs, then exit (no training)")
    args = ap.parse_args()

    ns, pretrain, critic, rl = build_stage_configs(
        args.bc_coef, args.adversarial, args.timesteps_override
    )

    if args.dry_run:
        _print_dry_run(args.bc_coef, args.adversarial, ns, pretrain, critic, rl)
        assert_invariants(args.bc_coef, args.adversarial, ns, pretrain, critic, rl)
        print("dry-run: all invariants OK")
        return 0

    # heavy imports deferred until we actually train (pull torch/pygame/envs)
    import train
    import pretrain_critic

    status_file = args.status_file or default_status_file(ns)
    totals = {"pretrain": pretrain.timesteps, "critic": critic.timesteps, "rl": rl.timesteps}
    progress.new_status(status_file, ns, args.bc_coef, args.adversarial, totals)
    progress.update_overall(status_file, state="running")

    stages = [
        ("pretrain", lambda: train.main(
            pretrain, extra_callbacks=[ProgressWriter(status_file, "pretrain", pretrain.timesteps)], notify=False)),
        ("critic", lambda: pretrain_critic.main(
            critic, extra_callbacks=[ProgressWriter(status_file, "critic", critic.timesteps)], notify=False)),
        ("rl", lambda: train.main(
            rl, extra_callbacks=[ProgressWriter(status_file, "rl", rl.timesteps)], notify=False)),
    ]

    for key, run in stages:
        print(f"\n===== [{ns}] stage: {key} =====")
        progress.set_stage(status_file, key, state="running")
        try:
            run()
        except Exception as e:  # noqa: BLE001
            progress.set_stage(status_file, key, state="failed")
            progress.update_overall(status_file, state="failed", error=f"{key}: {e}")
            traceback.print_exc()
            print(f"[{ns}] FAILED in stage {key}: {e}")
            return 1
        progress.set_stage(status_file, key, state="done")

    progress.update_overall(status_file, state="done")
    print(f"\n[{ns}] routine complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
