"""
scripts/ppo_bc_ablation/gen_eval_configs.py
============================================

One-off generator for the BC_Coef ablation eval configs. Emits 6 static YAMLs into
``scripts/eval/configs/`` — one per (eval test) x (checkpoint):

    bc_ablation_single_task_final.yaml    bc_ablation_single_task_best.yaml
    bc_ablation_switching_final.yaml      bc_ablation_switching_best.yaml
    bc_ablation_combination_final.yaml    bc_ablation_combination_best.yaml

Each lists the 30 ablation models (15 coefs x {uniform, adversarial}) plus the
three baselines the study compares against: ``rudin/comb_switching/2.0.0``,
``rudin_adv/comb_switching/2.0.0``, and ``hybrid``. The ``final`` configs point at
each model's ``rl/final.zip``; the ``best`` configs at ``rl/best/best_model.zip``.
Baselines stay at ``final.zip`` in both (they're fixed references).

Task blocks (5 single tasks / 7 switch endpoints / 2 combos) mirror the existing
``*_eval_5.yaml`` configs. Re-run after changing the sweep grid; the eval scripts
skip not-yet-trained models, so these can be committed before training finishes.

    python scripts/ppo_bc_ablation/gen_eval_configs.py
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "eval" / "configs"

COEFS = [round(0.1 * i, 2) for i in range(1, 16)]            # 0.10 .. 1.50
LINEAGES = [("uniform", False), ("adversarial", True)]

SINGLE_TASKS = """\
tasks:
  - { name: walk_forward,  label: "walk forward",  gait: [1, 0], cmd_vel_range: [0.0, 5.0],  cmd_tilt_range: [0.0, 0.0] }
  - { name: walk_backward, label: "walk backward", gait: [1, 0], cmd_vel_range: [-5.0, 0.0], cmd_tilt_range: [0.0, 0.0] }
  - { name: hop_forward,   label: "hop forward",   gait: [0, 1], cmd_vel_range: [0.0, 5.0],  cmd_tilt_range: [0.0, 0.0] }
  - { name: hop_backward,  label: "hop backward",  gait: [0, 1], cmd_vel_range: [-5.0, 0.0], cmd_tilt_range: [0.0, 0.0] }
  - { name: tilt,          label: "tilt",          gait: [1, 0], cmd_vel_range: [0.0, 0.0],  cmd_tilt_range: [-0.75, 0.75] }
"""

SWITCH_TASKS = """\
switch_tasks:
  - { name: walk_forward,  label: "walk forward",         gait: [1, 0], cmd_vel_range: [0.0, 5.0],  cmd_tilt_range: [0.0, 0.0] }
  - { name: walk_backward, label: "walk backward",        gait: [1, 0], cmd_vel_range: [-5.0, 0.0], cmd_tilt_range: [0.0, 0.0] }
  - { name: hop_forward,   label: "hop forward",          gait: [0, 1], cmd_vel_range: [0.0, 5.0],  cmd_tilt_range: [0.0, 0.0] }
  - { name: hop_backward,  label: "hop backward",         gait: [0, 1], cmd_vel_range: [-5.0, 0.0], cmd_tilt_range: [0.0, 0.0] }
  - { name: tilt,          label: "tilt",                 gait: [1, 0], cmd_vel_range: [0.0, 0.0],  cmd_tilt_range: [-0.75, 0.75] }
  - { name: walk_fwd+tilt, label: "walk forward + tilt",  gait: [1, 0], cmd_vel_range: [0.0, 5.0],  cmd_tilt_range: [-0.75, 0.75] }
  - { name: walk_bwd+tilt, label: "walk backward + tilt", gait: [1, 0], cmd_vel_range: [-5.0, 0.0], cmd_tilt_range: [-0.75, 0.75] }

exclude_pairs: null
"""

COMBINATIONS = """\
task_combinations:
  - { name: walk_fwd+tilt, label: "walk forward + tilt",  gait: [1, 0], cmd_vel_range: [0.0, 5.0],  cmd_tilt_range: [-0.75, 0.75] }
  - { name: walk_bwd+tilt, label: "walk backward + tilt", gait: [1, 0], cmd_vel_range: [-5.0, 0.0], cmd_tilt_range: [-0.75, 0.75] }
"""


def _model_entry(name: str, kind: str, ref: str | None, desc: str) -> str:
    return (f"  - name: {name}\n"
            f"    kind: {kind}\n"
            f"    ref: {'null' if ref is None else ref}\n"
            f'    desc: "{desc}"\n')


def models_block(checkpoint: str) -> str:
    """checkpoint: 'final' -> rl/final.zip ; 'best' -> rl/best/best_model.zip."""
    rel = "rl/final.zip" if checkpoint == "final" else "rl/best/best_model.zip"
    out = ["models:",
           "  # --- BC_Coef ablation (ours): 15 coefs x {uniform, adversarial} ---"]
    for lineage, adv in LINEAGES:
        for c in COEFS:
            ns = f"bc_ablation{'_adv' if adv else ''}/bc_{c:.2f}"
            desc = (f"BC_Coef ablation; pretrain BC coef={c:.2f}; {lineage} task selection; "
                    f"RL finetune on singles+combos union, fast switching (3s); "
                    f"BC pinned @ 0.1; {checkpoint} checkpoint")
            out.append(_model_entry(ns, "sb3", f"{ns}/{rel}", desc).rstrip("\n"))
    out.append("  # --- baselines ---")
    out.append(_model_entry(
        "rudin/comb_switching/2.0.0", "sb3", "rudin/comb_switching/2.0.0/final.zip",
        "Baseline (Rudin); pure-RL finetune on singles+combos union, fast switching (3s); "
        "no BC; uniform task selection").rstrip("\n"))
    out.append(_model_entry(
        "rudin_adv/comb_switching/2.0.0", "sb3", "rudin_adv/comb_switching/2.0.0/final.zip",
        "Baseline (Rudin); pure-RL finetune on singles+combos union, fast switching (3s); "
        "no BC; adversarial task selection").rstrip("\n"))
    out.append(_model_entry(
        "ppo_scratch/comb_switching/2.0.0", "sb3", "ppo_scratch/comb_switching/2.0.0/final.zip",
        "Baseline (from-scratch PPO, stock SB3); no BC/experts/pretraining/warm-start; trained "
        "one-shot on singles+combos union, fast switching (3s); 8.6M timesteps; uniform task "
        "selection").rstrip("\n"))
    out.append(_model_entry(
        "hybrid", "hybrid", None,
        "Hybrid oracle (gait routing) — topline reference baseline").rstrip("\n"))
    return "\n".join(out) + "\n"


def header(script: str, eval_name: str, checkpoint: str, body: str) -> str:
    ckpt_note = ("final RL checkpoint (rl/final.zip)" if checkpoint == "final"
                 else "best eval checkpoint (rl/best/best_model.zip)")
    return (
        f"# Experiment config for scripts/eval/{script}. Select with:\n"
        f"#     python scripts/eval/{script} {eval_name}\n"
        f"#\n"
        f"# BC_Coef ablation: the 30 swept models (15 coefs x uniform/adversarial) vs the\n"
        f"# rudin / rudin_adv / hybrid baselines, evaluated on the {ckpt_note}.\n"
        f"# GENERATED by scripts/ppo_bc_ablation/gen_eval_configs.py — edit there, not here.\n"
        f"# Episode counts mirror the *_eval_5 configs; lower them if 33 models is too heavy.\n\n"
        + body
    )


def single_task(checkpoint: str) -> tuple[str, str]:
    name = f"bc_ablation_single_task_{checkpoint}"
    body = (f"eval_name: {name}\n"
            f"task_scheme: gait\n"
            f"date_suffix: false\n"
            f"ep_time: 10\n"
            f"modulate_period: 3\n"
            f"episodes_per_task: 10000\n"
            f"episode_chunk: 100\n"
            f"seed_base: 42\n\n"
            f"{SINGLE_TASKS}\n"
            f"only_models: null\n\n"
            f"{models_block(checkpoint)}")
    return f"{name}.yaml", header("single_task_eval.py", name, checkpoint, body)


def switching(checkpoint: str) -> tuple[str, str]:
    name = f"bc_ablation_switching_{checkpoint}"
    body = (f"eval_name: {name}\n"
            f"task_scheme: gait\n\n"
            f"ep_time: 10\n"
            f"switch_fraction: 0.3333\n"
            f"episodes_per_pair: 1500\n"
            f"episode_chunk: 100\n"
            f"seed_base: 42\n\n"
            f"{SWITCH_TASKS}\n"
            f"only_models: null\n"
            f"date_suffix: false\n\n"
            f"{models_block(checkpoint)}")
    return f"{name}.yaml", header("task_switching_eval.py", name, checkpoint, body)


def combination(checkpoint: str) -> tuple[str, str]:
    name = f"bc_ablation_combination_{checkpoint}"
    body = (f"eval_name: {name}\n"
            f"task_scheme: gait\n"
            f"date_suffix: false\n"
            f"ep_time: 10\n"
            f"modulate_period: 3\n"
            f"episodes_per_combo: 10000\n"
            f"episode_chunk: 100\n"
            f"seed_base: 42\n\n"
            f"{COMBINATIONS}\n"
            f"only_models: null\n\n"
            f"{models_block(checkpoint)}")
    return f"{name}.yaml", header("task_combination_eval.py", name, checkpoint, body)


def main() -> None:
    builders = [single_task, switching, combination]
    written = []
    for build in builders:
        for checkpoint in ("final", "best"):
            fname, text = build(checkpoint)
            (CONFIG_DIR / fname).write_text(text)
            written.append(fname)
    print(f"wrote {len(written)} eval configs to {CONFIG_DIR}:")
    for f in written:
        print(f"  {f}")


if __name__ == "__main__":
    main()
