import os
import sys
from gymnasium import make
import numpy as np
import pygame
from stable_baselines3 import PPO

from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from utils.paths import MODELS_DIR
from wrappers.bipedal_walker.distill_env import DistillEnv
from wrappers.bipedal_walker.sit_distill_reward import SitDistillReward

from gymnasium import Wrapper


class MagnitudeInitVelWrapper(Wrapper):
    def __init__(self, env, mag_range, sign_bias: float = 0.5):
        super().__init__(env)
        self._mag = mag_range
        self._sign_bias = sign_bias

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        env_u = self.unwrapped
        sign = 1.0 if np.random.random() < self._sign_bias else -1.0
        mag = float(np.random.uniform(*self._mag))
        env_u.hull.linearVelocity = (sign * mag, env_u.hull.linearVelocity.y)
        env_u.hull.awake = True
        return obs, info


# =========================================
# Composite: route by init vel sign.
# fwd policy for positive init vel, back policy for negative.
# Override either via CLI: play.py FWD_MODEL [BACK_MODEL]
# =========================================
FWD_MODEL = "../final_composites/not_shin_flat/fwd_v15a_14a_resume_long_5M"
BACK_MODEL = "../final_composites/not_shin_flat/back_v21a_back_v15a_recipe_5.2M"
MAG_RANGE = (3.0, 5.0)
SIGN_BIAS = 0.5
# =========================================

_sim_paused = False
_sim_step = False
_sim_res = False


def main():
    global _sim_paused, _sim_step, _sim_res
    fwd_rel = sys.argv[1] if len(sys.argv) > 1 else FWD_MODEL
    back_rel = sys.argv[2] if len(sys.argv) > 2 else BACK_MODEL
    fwd_path = MODELS_DIR / f"{fwd_rel}.zip"
    back_path = MODELS_DIR / f"{back_rel}.zip"
    for p in (fwd_path, back_path):
        if not p.exists():
            raise SystemExit(f"model not found: {p}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    env = make("BipedalWalker-v3", render_mode="rgb_array")
    env = DistillEnv(env, ep_time=15, tasks={1: "sit"})
    env.set_task(1)
    env.config_hull_reset(
        x_range=(0.0, 40.0), y_range=(0.0, 0.3),
        rot_range=(-0.2, 0.2),
        vel_x_range=(0.0, 0.0), vel_y_range=(0.0, 0.0),
    )
    env.config_joint_reset(
        hip_range=(-0.6, 0.6),
        knee_range=(-1.4, -0.1),
        joint_vel_range=(-0.2, 0.2),
    )
    env.config_cmd_vel(sample_range=(0.0, 0.0), switch_time=1000.0, interp_time=0.0, zero_prob=1.0)
    env = MagnitudeInitVelWrapper(env, MAG_RANGE, sign_bias=SIGN_BIAS)
    env = SitDistillReward(env)

    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    clock = pygame.time.Clock()

    print(f"loading fwd:  {fwd_path}")
    print(f"loading back: {back_path}")
    fwd_model = PPO.load(fwd_path, env=None, device="cpu")
    back_model = PPO.load(back_path, env=None, device="cpu")

    obs, _ = env.reset()
    # pick policy based on init vel sign
    init_vx = float(env.unwrapped.hull.linearVelocity.x)
    active = fwd_model if init_vx > 0 else back_model
    active_name = "FWD" if init_vx > 0 else "BACK"

    pygame.font.init()
    font = pygame.font.SysFont("Courier New", 13, bold=True)

    n_eps = 0
    n_survived = 0
    n_fwd_eps = n_fwd_surv = 0
    n_back_eps = n_back_surv = 0
    last_diag: dict = {}
    last_init_vx = init_vx

    def render(extra_lines: list[str]):
        frame = env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            screen.blit(surf, (0, 0))
            y = 10
            for line in extra_lines:
                screen.blit(font.render(line, True, (255, 255, 255), (0, 0, 0)), (10, y))
                y += 15
            pygame.display.flip()
        clock.tick(50)

    def fmt_lines(d):
        if not d:
            return []
        ok_bits = "".join(
            c if d.get(f"ok_{name}", 0) else "."
            for c, name in zip("KSCUJI", ["knees", "shin", "contacts", "upright", "still", "jitter"])
        )
        fpct = f"{100*n_fwd_surv/n_fwd_eps:.0f}%" if n_fwd_eps else "-"
        bpct = f"{100*n_back_surv/n_back_eps:.0f}%" if n_back_eps else "-"
        return [
            f"[{active_name}]  init_vx={last_init_vx:+.2f}",
            f"sit_ok={int(d.get('sit_ok',0))}  flags[{ok_bits}]",
            f"hull  h={d.get('hull_height',0):+.2f}  ang={d.get('hull_ang_deg',0):+.1f}d",
            f"vel   x={d.get('hull_vel_x',0):+.2f} y={d.get('hull_vel_y',0):+.2f}",
            f"shin  1={d.get('shin1_world',0):+.2f}  2={d.get('shin2_world',0):+.2f}  kneel={int(d.get('kneel_leg',0))+1}",
            f"hip   1={d.get('hip1',0):+.2f}  2={d.get('hip2',0):+.2f}",
            f"knee  1={d.get('knee1',0):+.2f}  2={d.get('knee2',0):+.2f}",
            f"jitter={d.get('action_delta_raw',0):.3f}  a_l2={d.get('action_l2_raw',0):.3f}",
            f"ep surv {n_survived}/{n_eps}" + (f" ({100*n_survived/n_eps:.0f}%)" if n_eps else ""),
            f"fwd={fpct} ({n_fwd_surv}/{n_fwd_eps})  back={bpct} ({n_back_surv}/{n_back_eps})",
        ]

    while True:
        pygame.event.pump()
        if _sim_res:
            _sim_res = False
            obs, _ = env.reset()
            init_vx = float(env.unwrapped.hull.linearVelocity.x)
            active = fwd_model if init_vx > 0 else back_model
            active_name = "FWD" if init_vx > 0 else "BACK"
            last_init_vx = init_vx
            render(fmt_lines(last_diag))
            continue
        if _sim_paused:
            if not _sim_step:
                continue
            _sim_step = False

        action, _ = active.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        last_diag = info.get("diag", last_diag)
        render(fmt_lines(last_diag))
        if term or trunc:
            n_eps += 1
            if active_name == "FWD": n_fwd_eps += 1
            else: n_back_eps += 1
            if "episode_end" in info and info["episode_end"]["survived"]:
                n_survived += 1
                if active_name == "FWD": n_fwd_surv += 1
                else: n_back_surv += 1
            _sim_res = True


def on_press(key):
    global _sim_paused, _sim_step, _sim_res
    if isinstance(key, KeyCode):
        k = key.char
    elif isinstance(key, Key):
        k = key.name
    else:
        return
    if k == "space":
        _sim_paused = not _sim_paused
    elif k == "s":
        _sim_step = True
    elif k == "r":
        _sim_res = True
    elif k == "q":
        os._exit(0)


if __name__ == "__main__":
    main()
