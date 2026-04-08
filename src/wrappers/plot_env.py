import signal
from typing import Any, SupportsFloat
import sys, subprocess, os

from gymnasium import Env, Wrapper
from gymnasium.core import ActType, ObsType

import matplotlib.pyplot as plt


WINDOW = 50


class Plotter(Wrapper):
    """
    Wrapper that shows a live MLP dashboard of BipedalWalker observations.
    Make sure this is the outermost wrapper!!

    Opens a 4x2 grid of scrolling time-series plots covering hull kinematics,
    joint positions/velocities, ground contact, and lidar readings. The plot
    window updates every step and keeps the last `window` timesteps visible.
    """

    def __init__(self, env: Env[ObsType, ActType], window: int = WINDOW):
        super().__init__(env)
        
        self._t = 0
        self._window = window

        plt.ion()
        def handler(sig, frame):
            plt.close('all')
            raise SystemExit(0)
        # Restore default SIGINT so Ctrl+C works while matplotlib processes events
        signal.signal(signal.SIGINT, handler)
        
        self._fig, self._axes = plt.subplots(
            5, 2,
            sharex="all",
            squeeze=False,
            figsize=(8, 7.5)
        )

        self._fig.tight_layout(pad=2.5)

        titles = [
            "hull_ang", "ang_vel",
            "vel_x", "vel_y",
            "joint_pos", "joint_vel",
            "contact", "lidar",
            "reward", None,
        ]
        for i, title in enumerate(titles):
            r, c = divmod(i, 2)
            if title is None:
                self._axes[r, c].set_visible(False)
            else:
                self._axes[r, c].set_title(title)

        # scalar obs
        self._lines_scalar = []
        for i in range(4):
            r, c = divmod(i, 2)
            line, = self._axes[r, c].plot([], [])
            self._lines_scalar.append(line)

        # joint pos
        self._lines_joint_pos = [
            self._axes[2, 0].plot([], [], label=f"hip_1")[0],
            self._axes[2, 0].plot([], [], label=f"hip_2")[0],
            self._axes[2, 0].plot([], [], label=f"knee_1")[0],
            self._axes[2, 0].plot([], [], label=f"knee_2")[0]
        ]
        self._axes[2, 0].legend(fontsize="x-small", loc="upper right")

        # joint vel
        self._lines_joint_vel = [
            self._axes[2, 1].plot([], [], label=f"hip_1")[0],
            self._axes[2, 1].plot([], [], label=f"hip_2")[0],
            self._axes[2, 1].plot([], [], label=f"knee_1")[0],
            self._axes[2, 1].plot([], [], label=f"knee_2")[0]
        ]
        self._axes[2, 1].legend(fontsize="x-small", loc="upper right")

        # contact
        self._lines_contact = [
            self._axes[3, 0].plot([], [], label=f"leg_{i+1}")[0] for i in range(2)
        ]
        self._axes[3, 0].legend(fontsize="x-small", loc="upper right")

        # lidars
        self._lines_lidar = [
            self._axes[3, 1].plot([], [], label=str(i), linewidth=0.8)[0] for i in range(10)
        ]
        self._axes[3, 1].legend(fontsize="x-small", loc="upper right", ncol=2)

        # reward
        self._line_reward, = self._axes[4, 0].plot([], [])

        # data buffers
        self._ts: list[int] = []
        self._data_scalar  = [[] for _ in range(4)]
        self._data_joint_pos    = [[] for _ in range(4)]
        self._data_joint_vel    = [[] for _ in range(4)]
        self._data_contact = [[] for _ in range(2)]
        self._data_lidar   = [[] for _ in range(10)]
        self._data_reward: list[float] = []
        self._plot_shown = False
        

    def _raise_pygame_window(self):
        """
        Bring the pygame render window to the foreground (macOS only).

        Uses AppleScript to raise every window belonging to this process that
        is not a matplotlib Figure, so the pygame render window comes to the
        front after the plot is first shown.
        """
        # macos specific, bring pygame window to the front
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
            subprocess.run(['osascript', '-e', script], check=False)
        except Exception:
            pass


    def _update_plots(self, obs, reward):
        """
        Append the latest observation to all data buffers and redraw the dashboard.

        On the first call, makes the matplotlib window visible and then raises
        the pygame window so focus returns to the render view.

        Args:
            obs: The raw 24-element BipedalWalker observation vector from the
                 current step.
        """
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

        # scalars
        for i in range(4):
            _append(self._data_scalar[i], obs[i])
            self._lines_scalar[i].set_data(ts, self._data_scalar[i])
            r, c = divmod(i, 2)
            self._axes[r, c].relim()
            self._axes[r, c].autoscale_view()

        # joint pos
        _append(self._data_joint_pos[0], obs[4]) # h1
        _append(self._data_joint_pos[1], obs[9]) # h2
        _append(self._data_joint_pos[2], obs[6]) # k1
        _append(self._data_joint_pos[3], obs[11]) # k2
        
        for i in range(4):
            self._lines_joint_pos[i].set_data(ts, self._data_joint_pos[i])
        
        self._axes[2, 0].relim()
        self._axes[2, 0].autoscale_view()

        # joint vel
        _append(self._data_joint_vel[0], obs[5]) # h1
        _append(self._data_joint_vel[1], obs[10]) # h2
        _append(self._data_joint_vel[2], obs[7]) # k1
        _append(self._data_joint_vel[3], obs[12]) # k2
        
        for i in range(4):
            self._lines_joint_vel[i].set_data(ts, self._data_joint_vel[i])
            
        self._axes[2, 1].relim()
        self._axes[2, 1].autoscale_view()

        # contact
        _append(self._data_contact[0], obs[8]) # l1
        _append(self._data_contact[1], obs[13]) # l2
        
        for i in range(2):
            self._lines_contact[i].set_data(ts, self._data_contact[i])    
        
        self._axes[3, 0].relim()
        self._axes[3, 0].autoscale_view()

        # lidars
        for i in range(10):
            _append(self._data_lidar[i], obs[14 + i])
            self._lines_lidar[i].set_data(ts, self._data_lidar[i])
        self._axes[3, 1].relim()
        self._axes[3, 1].autoscale_view()

        # reward
        _append(self._data_reward, reward)
        self._line_reward.set_data(ts, self._data_reward)
        self._axes[4, 0].relim()
        self._axes[4, 0].autoscale_view()

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Step the environment, update plots, and return the transition.
        """

        s = super().step(action)
        
        self._t += 1
        self._last_s = s;

        self._update_plots(s[0], s[1])

        return s
