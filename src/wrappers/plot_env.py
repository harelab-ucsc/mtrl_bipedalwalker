import signal
from typing import Any, SupportsFloat
import sys, subprocess, os

from gymnasium import Env, Wrapper
from gymnasium.core import ActType, ObsType

import matplotlib.pyplot as plt


WINDOW = 50


def _screen_size() -> tuple[int, int]:
    """Return logical screen dimensions without initialising a GUI toolkit."""
    if sys.platform == "darwin":
        # tkinter conflicts with SDL's NSApplication on macOS — use CoreGraphics
        try:
            import ctypes, ctypes.util

            cg = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreGraphics"))

            class _CGPoint(ctypes.Structure):
                _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

            class _CGSize(ctypes.Structure):
                _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]

            class _CGRect(ctypes.Structure):
                _fields_ = [("origin", _CGPoint), ("size", _CGSize)]

            cg.CGMainDisplayID.restype = ctypes.c_uint32
            cg.CGDisplayBounds.restype = _CGRect
            cg.CGDisplayBounds.argtypes = [ctypes.c_uint32]
            bounds = cg.CGDisplayBounds(cg.CGMainDisplayID())
            return int(bounds.size.width), int(bounds.size.height)
        except Exception:
            pass
    else:
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            return sw, sh
        except Exception:
            pass
    return 1920, 1080  # safe fallback


def _fit_to_screen(w: float, h: float, margin: float = 0.88) -> tuple[float, float]:
    """Scale (w, h) in inches down so the figure fits within `margin` of the screen."""
    sw, sh = _screen_size()
    dpi = plt.rcParams.get("figure.dpi", 100)
    scale = min(sw / dpi * margin / w, sh / dpi * margin / h, 1.0)
    return w * scale, h * scale


class Plotter(Wrapper):
    """
    Wrapper that shows a live dashboard of BipedalWalker proprioceptive observations.
    Make sure this is the outermost wrapper!!

    Automatically detects whether the observation contains lidar (obs dim >= 24).
    Without lidar the panel is dropped and reward takes its place. The plot window
    updates every step and keeps the last `window` timesteps visible.
    """

    def __init__(self, env: Env[ObsType, ActType], window: int = WINDOW):
        super().__init__(env)

        self._t = 0
        self._window = window

        obs_dim = env.observation_space.shape[0]  # type: ignore[index]
        self._has_lidar = obs_dim >= 24

        plt.ion()

        def handler(sig, frame):
            plt.close("all")
            raise SystemExit(0)

        signal.signal(signal.SIGINT, handler)

        if self._has_lidar:
            self._fig, self._axes = plt.subplots(
                5, 2, sharex="all", squeeze=False, figsize=_fit_to_screen(8, 7.5)
            )
            titles = [
                "hull_ang", "ang_vel",
                "vel_x", "vel_y",
                "joint_pos", "joint_vel",
                "contact", "lidar",
                "reward", None,
            ]
        else:
            self._fig, self._axes = plt.subplots(
                4, 2, sharex="all", squeeze=False, figsize=_fit_to_screen(8, 6.0)
            )
            titles = [
                "hull_ang", "ang_vel",
                "vel_x", "vel_y",
                "joint_pos", "joint_vel",
                "contact", "reward",
            ]

        self._fig.tight_layout(pad=2.5)

        for i, title in enumerate(titles):
            r, c = divmod(i, 2)
            if title is None:
                self._axes[r, c].set_visible(False)
            else:
                self._axes[r, c].set_title(title)

        # scalar obs: hull_ang, ang_vel, vel_x, vel_y
        self._lines_scalar = []
        for i in range(4):
            r, c = divmod(i, 2)
            line, = self._axes[r, c].plot([], [])
            self._lines_scalar.append(line)

        # joint pos
        self._lines_joint_pos = [
            self._axes[2, 0].plot([], [], label="hip_1")[0],
            self._axes[2, 0].plot([], [], label="hip_2")[0],
            self._axes[2, 0].plot([], [], label="knee_1")[0],
            self._axes[2, 0].plot([], [], label="knee_2")[0],
        ]
        self._axes[2, 0].legend(fontsize="x-small", loc="upper right")

        # joint vel
        self._lines_joint_vel = [
            self._axes[2, 1].plot([], [], label="hip_1")[0],
            self._axes[2, 1].plot([], [], label="hip_2")[0],
            self._axes[2, 1].plot([], [], label="knee_1")[0],
            self._axes[2, 1].plot([], [], label="knee_2")[0],
        ]
        self._axes[2, 1].legend(fontsize="x-small", loc="upper right")

        # contact
        self._lines_contact = [
            self._axes[3, 0].plot([], [], label=f"leg_{i+1}")[0] for i in range(2)
        ]
        self._axes[3, 0].legend(fontsize="x-small", loc="upper right")

        if self._has_lidar:
            # lidar panel at (3, 1)
            self._lines_lidar = [
                self._axes[3, 1].plot([], [], label=str(i), linewidth=0.8)[0]
                for i in range(10)
            ]
            self._axes[3, 1].legend(fontsize="x-small", loc="upper right", ncol=2)
            # reward at (4, 0)
            self._line_reward, = self._axes[4, 0].plot([], [])
            self._reward_ax = self._axes[4, 0]
        else:
            # no lidar panel; reward at (3, 1)
            self._lines_lidar = []
            self._line_reward, = self._axes[3, 1].plot([], [])
            self._reward_ax = self._axes[3, 1]

        # data buffers
        self._ts: list[int] = []
        self._data_scalar = [[] for _ in range(4)]
        self._data_joint_pos = [[] for _ in range(4)]
        self._data_joint_vel = [[] for _ in range(4)]
        self._data_contact = [[] for _ in range(2)]
        self._data_lidar = [[] for _ in range(10)] if self._has_lidar else []
        self._data_reward: list[float] = []
        self._plot_shown = False

    def _raise_pygame_window(self):
        """Bring the pygame render window to the foreground (macOS only)."""
        if sys.platform != "darwin":
            return
        try:
            script = (
                'tell application "System Events"\n'
                f'    set proc to first process whose unix id is {os.getpid()}\n'
                '    set frontmost of proc to true\n'
                '    repeat with w in (every window of proc)\n'
                '        if name of w does not contain "Figure" then\n'
                '            perform action "AXRaise" of w\n'
                '        end if\n'
                '    end repeat\n'
                'end tell'
            )
            subprocess.run(["osascript", "-e", script], check=False)
        except Exception:
            pass

    def _update_plots(self, obs, reward):
        if not self._plot_shown:
            plt.show(block=False)
            plt.pause(0.01)
            self._raise_pygame_window()
            self._plot_shown = True

        self._ts.append(self._t)
        if len(self._ts) > self._window:
            self._ts = self._ts[-self._window:]
        ts = self._ts

        def _append(buf, val):
            buf.append(float(val))
            if len(buf) > self._window:
                del buf[0]

        # scalars: indices 0-3 are the same in both proprio and full obs
        for i in range(4):
            _append(self._data_scalar[i], obs[i])
            self._lines_scalar[i].set_data(ts, self._data_scalar[i])
            r, c = divmod(i, 2)
            self._axes[r, c].relim()
            self._axes[r, c].autoscale_view()

        # joint pos: h1=4, k1=6, h2=9, k2=11 — same in both layouts
        _append(self._data_joint_pos[0], obs[4])   # hip_1
        _append(self._data_joint_pos[1], obs[9])   # hip_2
        _append(self._data_joint_pos[2], obs[6])   # knee_1
        _append(self._data_joint_pos[3], obs[11])  # knee_2
        for i in range(4):
            self._lines_joint_pos[i].set_data(ts, self._data_joint_pos[i])
        self._axes[2, 0].relim()
        self._axes[2, 0].autoscale_view()

        # joint vel: h1=5, k1=7, h2=10, k2=12 — same in both layouts
        _append(self._data_joint_vel[0], obs[5])   # hip_1
        _append(self._data_joint_vel[1], obs[10])  # hip_2
        _append(self._data_joint_vel[2], obs[7])   # knee_1
        _append(self._data_joint_vel[3], obs[12])  # knee_2
        for i in range(4):
            self._lines_joint_vel[i].set_data(ts, self._data_joint_vel[i])
        self._axes[2, 1].relim()
        self._axes[2, 1].autoscale_view()

        # contact: leg1=8, leg2=13 — same in both layouts
        _append(self._data_contact[0], obs[8])
        _append(self._data_contact[1], obs[13])
        for i in range(2):
            self._lines_contact[i].set_data(ts, self._data_contact[i])
        self._axes[3, 0].relim()
        self._axes[3, 0].autoscale_view()

        if self._has_lidar:
            for i in range(10):
                _append(self._data_lidar[i], obs[14 + i])
                self._lines_lidar[i].set_data(ts, self._data_lidar[i])
            self._axes[3, 1].relim()
            self._axes[3, 1].autoscale_view()

        # reward
        _append(self._data_reward, reward)
        self._line_reward.set_data(ts, self._data_reward)
        self._reward_ax.relim()
        self._reward_ax.autoscale_view()

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        s = super().step(action)
        self._t += 1
        self._update_plots(s[0], s[1])
        return s
