"""
Verification for the redesigned finetuning reward (rl_finetune_rewards.py).

Run:  PYTHONPATH=src python scripts/eval/verify_rewards.py

Checks:
  A. Kernel gradient — track() keeps usable gradient in the tails (Lorentzian),
     unlike a Gaussian.
  B. Per-task ceiling / scale consistency — an ideal tracker scores a comparable
     r_task across walk / tilt / flamingo / walk+tilt / flamingo+tilt, all in [0,1].
  C. Regularization is bounded and ~0 (+alive) for an ideal, smooth, upright pose.
  D. Real-env integration smoke — RlFTEnv steps through task switches with finite,
     bounded rewards and no spurious gait spike at a switch.
"""

import types

import numpy as np

from mdp.bipedal_walker.rl_finetune_rewards import (
    ALIVE_BONUS,
    K_VEL,
    SIGMA_VEL,
    TARGET_HEIGHT,
    W_ANG,
    W_GAIT,
    W_VEL,
    RewardState,
    compositional_rew,
    track,
)


def _mock_env(vx=0.0, vy=0.0, ang=0.0, ang_vel=0.0, x=30.0, height=None):
    """Minimal stand-in for a BipedalWalker exposing only what the reward reads.
    Ground is flat at y=0, so height_above_ground == hull.position.y."""
    if height is None:
        height = TARGET_HEIGHT
    hull = types.SimpleNamespace(
        angularVelocity=ang_vel,
        linearVelocity=types.SimpleNamespace(x=vx, y=vy),
        position=types.SimpleNamespace(x=x, y=height),
        angle=ang,
    )
    return types.SimpleNamespace(
        hull=hull,
        terrain_x=np.array([0.0, 1000.0]),
        terrain_y=np.array([0.0, 0.0]),
    )


def _base_obs(leg1=0, leg2=0):
    obs = np.zeros(14)
    obs[8] = leg1
    obs[13] = leg2
    return obs


def check_A_kernel():
    print("\n=== A. kernel gradient (velocity channel) ===")
    print(f"{'err':>6}  {'track':>8}  {'gaussian':>9}")
    for err in [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]:
        g = float(np.exp(-(err**2) / (2 * SIGMA_VEL**2)))
        print(f"{err:6.2f}  {track(err, SIGMA_VEL, K_VEL):8.4f}  {g:9.6f}")
    # finite-difference gradient magnitude far out (err=5)
    d = 1e-3
    grad_track = abs(track(5 + d, SIGMA_VEL, K_VEL) - track(5 - d, SIGMA_VEL, K_VEL)) / (2 * d)
    grad_gauss = abs(
        np.exp(-((5 + d) ** 2) / (2 * SIGMA_VEL**2)) - np.exp(-((5 - d) ** 2) / (2 * SIGMA_VEL**2))
    ) / (2 * d)
    print(f"|grad| at err=5:  track={grad_track:.3e}   gaussian={grad_gauss:.3e}")
    assert abs(track(0.0, SIGMA_VEL, K_VEL) - 1.0) < 1e-9, "track(0) must be 1"
    assert grad_track > 10 * grad_gauss, "Lorentzian tail should dominate Gaussian gradient"
    # strictly decreasing in |err|
    vals = [track(e, SIGMA_VEL, K_VEL) for e in [0.0, 0.5, 1.0, 2.0, 5.0]]
    assert all(a > b for a, b in zip(vals, vals[1:])), "track must be monotone decreasing"
    print("OK: track(0)=1, monotone, tail gradient >> Gaussian.")


def _ideal_rollout(task_bits, cmd_vel, cmd_tilt, contact_pattern, n=150, mode=None):
    """Run an ideal-tracking rollout (vx=cmd_vel, ang=cmd_tilt, smooth, upright-for-tilt).
    contact_pattern(step) -> (leg1, leg2). Returns dict of per-step r_task / reg / total.
    ``mode`` (walk/hop/quiet) is passed through to the reward — None lets it derive the
    legacy onehot mode from task_bits; gait callers pass an explicit reward_mode(...)."""
    st = RewardState(prev_vel_x=cmd_vel, prev_vel_y=0.0, prev_accel_x=0.0, prev_accel_y=0.0)
    height = TARGET_HEIGHT * float(np.cos(cmd_tilt))  # perfect tilt-aware height
    r_task_hist, reg_hist, total_hist = [], [], []
    for step in range(n):
        leg1, leg2 = contact_pattern(step)
        env = _mock_env(vx=cmd_vel, ang=cmd_tilt, ang_vel=0.0, height=height)
        obs = _base_obs(leg1, leg2)
        total, comps, _raw, _w, st = compositional_rew(
            env, obs, terminated=False, state=st,
            task_bits=task_bits, cmd_vel=cmd_vel, cmd_tilt=cmd_tilt,
            mode=mode,
            enable_task_reward=True,
        )
        r_task = sum(v for k, v in comps.items() if k.startswith("track_"))
        reg = sum(v for k, v in comps.items() if not k.startswith("track_"))
        r_task_hist.append(r_task)
        reg_hist.append(reg)
        total_hist.append(total)
    return {
        "r_task_max": max(r_task_hist),
        "r_task_mean": float(np.mean(r_task_hist)),
        "reg_min": min(reg_hist),
        "reg_max": max(reg_hist),
        "total_max": max(total_hist),
    }


def _alt(step, stride=16):
    """Alternating gait: one leg down per stride, switching each stride."""
    return (1, 0) if (step // stride) % 2 == 0 else (0, 1)


def _hop(step, stride=16):
    """Single-leg hop: leg1 lands every stride (airborne in between), leg2 never."""
    return (1 if step % stride < stride // 2 else 0, 0)


def _both(step):
    return (1, 1)


def check_BC_scale():
    print("\n=== B/C. per-task ceiling, scale consistency, bounded reg ===")
    tasks = [
        ("walk",          (1, 0, 0),  2.0, 0.0, _alt),
        ("tilt",          (0, 0, 1),  0.0, 0.4, _both),
        ("flamingo",      (0, 1, 0),  0.0, 0.0, _hop),
        ("walk+tilt",     (1, 0, 1),  2.0, 0.4, _alt),
        ("flamingo+tilt", (0, 1, 1),  0.0, 0.4, _hop),
    ]
    print(f"{'task':>14}  {'r_task_max':>10}  {'r_task_mean':>11}  {'reg[min,max]':>16}")
    ceilings = []
    for name, bits, v, t, pat in tasks:
        s = _ideal_rollout(bits, v, t, pat)
        ceilings.append(s["r_task_max"])
        print(f"{name:>14}  {s['r_task_max']:10.3f}  {s['r_task_mean']:11.3f}  "
              f"[{s['reg_min']:+.3f},{s['reg_max']:+.3f}]")
        assert -1e-9 <= s["r_task_max"] <= 1.0 + 1e-6, f"{name}: r_task out of [0,1]"
        assert s["reg_min"] >= -1.0, f"{name}: reg below bound"
        # ideal pose -> reg should be just the alive bonus (no penalties fire)
        assert abs(s["reg_max"] - ALIVE_BONUS) < 1e-6, f"{name}: reg_max != alive bonus"
    spread = max(ceilings) - min(ceilings)
    print(f"ceiling spread across tasks = {spread:.3f}  (track-only ceiling = {W_VEL + W_ANG:.2f}, "
          f"+gait up to {W_GAIT:.2f})")
    assert spread < 0.25, "per-task ceilings should be comparable"
    print("OK: every task tops out in [0,1] with a comparable ceiling; reg bounded.")


def check_D_env_smoke():
    print("\n=== D. real-env integration smoke (RlFTEnv) ===")
    try:
        from gymnasium import make
        from wrappers.ppo_bc.ppo_bc_env import RlFTEnv
    except Exception as e:  # pragma: no cover
        print(f"SKIP (env import failed: {e})")
        return

    from mdp.bipedal_walker.tasks import ONEHOT

    raw = make("BipedalWalker-v3")
    env = RlFTEnv(
        raw,
        ep_time=10,
        cmd_switching_time=(2.0, 2.0),
        task_switching_time=1.0,  # force frequent task switches
        # with replacement: the fast switching here would otherwise demand more
        # distinct draws/episode than there are tasks (the without-replacement default).
        task_switch_replacement=True,
        # onehot tuples → run this smoke under the onehot scheme.
        allowed_task_mixing=[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1)],
        use_rew_for_individual_tasks=True,
        manual_ctrl_mode=False,
        task_scheme=ONEHOT,
    )
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs))

    prev_task = None
    max_gait_after_switch = 0.0
    n_switches = 0
    rewards = []
    for _ in range(600):
        obs, rew, term, trunc, info = env.step(env.action_space.sample() * 0.0)
        assert np.isfinite(rew), "reward must be finite"
        assert np.all(np.isfinite(obs)), "obs must be finite"
        assert "reward_terms" in info and "task_name" in info
        rewards.append(rew)
        # Detect switches via info["task"], which labels the task used for THIS
        # step's reward. The switch (and gait-phase reset) is applied at the END of
        # the prior step, so the step where info["task"] first changes is the
        # incoming task's FIRST reward step — exactly where a stale cadence would
        # show up if the phase weren't reset.
        cur_task = tuple(info["task"])
        if prev_task is not None and cur_task != prev_task:
            n_switches += 1
            # A RHYTHMIC incoming task (walk/flamingo) must show no carried-over
            # cadence on its first step. Quiet mode (tilt-only/idle) is excluded:
            # it legitimately reads W_GAIT the instant both feet are planted.
            if cur_task[0] or cur_task[1]:
                g = info["reward_terms"].get("track_gait", 0.0)
                max_gait_after_switch = max(max_gait_after_switch, g)
        prev_task = cur_task
        if term or trunc:
            obs, _ = env.reset()
            prev_task = None
    env.close()
    print(f"steps=600  task_switches={n_switches}  "
          f"reward[min,max]=[{min(rewards):.2f},{max(rewards):.2f}]  "
          f"max track_gait on switch-step={max_gait_after_switch:.3f}")
    assert all(np.isfinite(rewards))
    assert max_gait_after_switch < 0.2, "spurious gait bonus right after a task switch"
    print("OK: finite/bounded rewards across switches; no stale-gait spike.")


def check_E_hop_both_down():
    print("\n=== E. hop both-legs-down penalty (mode='hop') ===")
    from mdp.bipedal_walker.rl_finetune_rewards import W_HOP_BOTH_DOWN

    # hop mode, ideal vel/tilt tracking. The only difference is the gait:
    # _both = parked on both legs (the cheat), _hop = proper single-leg hop.
    cheat = _ideal_rollout((0, 1, 0), 0.0, 0.4, _both, mode="hop")
    hop = _ideal_rollout((0, 1, 0), 0.0, 0.4, _hop, mode="hop")
    print(f"{'stance':>14}  {'total_max':>10}")
    print(f"{'both-down':>14}  {cheat['total_max']:10.3f}")
    print(f"{'one-leg hop':>14}  {hop['total_max']:10.3f}")
    assert W_HOP_BOTH_DOWN < 0, "both-down penalty weight must be negative"
    # parking on both legs farms only vel+tilt tracking, then eats the penalty —
    # the best step it can manage is net-negative.
    assert cheat["total_max"] < 0.0, "both-legs-down in hop must be net-negative"
    # hopping never lands both feet, so the penalty never fires; it must clearly win.
    assert hop["total_max"] > cheat["total_max"] + 0.5, "hop must clearly beat both-down"
    print("OK: parking on both legs is penalized net-negative; single-leg hop dominates.")


def check_F_gait_scale():
    """Gait scheme: verify the per-task tracking matrix (the bit the redesign hinges
    on). Velocity is tracked for EVERY task incl. directional hops; tilt is tracked for
    tilt / walk+tilt and held upright (0) for walk / hop; the gait term rewards
    alternation (walk) / same-leg hop (hop) / planted feet (quiet)."""
    from mdp.bipedal_walker.tasks import GAIT, reward_mode

    print("\n=== F. gait per-task tracking matrix ===")
    # (name, gait_bits, cmd_vel, cmd_tilt, contact_pattern)
    cases = [
        ("walk_forward",  (1, 0, 0),  2.0, 0.0, _alt),
        ("walk_backward", (1, 0, 0), -2.0, 0.0, _alt),
        ("hop_forward",   (0, 1, 0),  2.0, 0.0, _hop),
        ("hop_in_place",  (0, 1, 0),  0.0, 0.0, _hop),
        ("tilt",          (1, 0, 0),  0.0, 0.4, _both),
        ("walk_fwd+tilt", (1, 0, 0),  2.0, 0.4, _alt),
    ]
    print(f"{'task':>14}  {'mode':>6}  {'r_task_max':>10}  {'r_task_mean':>11}")
    ceilings = []
    for name, bits, v, t, pat in cases:
        mode = reward_mode(bits, v, GAIT)
        s = _ideal_rollout(bits, v, t, pat, mode=mode)
        ceilings.append(s["r_task_max"])
        print(f"{name:>14}  {mode:>6}  {s['r_task_max']:10.3f}  {s['r_task_mean']:11.3f}")
        assert -1e-9 <= s["r_task_max"] <= 1.0 + 1e-6, f"{name}: r_task out of [0,1]"
        assert abs(s["reg_max"] - ALIVE_BONUS) < 1e-6, f"{name}: reg_max != alive bonus"
    # moving hop must score meaningfully (velocity IS tracked for one-leg gait) — a
    # near-zero ceiling would mean velocity got masked out.
    hop_fwd = _ideal_rollout((0, 1, 0), 2.0, 0.0, _hop, mode="hop")
    assert hop_fwd["r_task_mean"] > 0.2, "moving hop must track its commanded velocity"
    spread = max(ceilings) - min(ceilings)
    print(f"ceiling spread across gait tasks = {spread:.3f}")
    assert spread < 0.25, "per-task gait ceilings should be comparable"
    print("OK: gait tasks track vel/tilt per the matrix; comparable ceilings in [0,1].")


if __name__ == "__main__":
    check_A_kernel()
    check_BC_scale()
    check_D_env_smoke()
    check_E_hop_both_down()
    check_F_gait_scale()
    print("\nAll reward verification checks passed.")
