import math
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


class RewardPlotter(Wrapper):
    """
    Wrapper that shows a live per-term reward breakdown dashboard.
    Make sure this is the outermost wrapper!!

    One scrolling subplot per reward term plus one for the total reward.
    Below each subplot a text line shows:
        raw value | scaled value (raw × weight) | |contribution|% of |total|

    Requires the inner environment to populate info with:
        info["reward_terms"]   — {name: scaled_value}
        info["reward_raw"]     — {name: raw_value}
        info["reward_weights"] — {name: weight}

    All custom reward wrappers in this project emit these automatically.
    """

    def __init__(self, env: Env[ObsType, ActType], window: int = WINDOW):
        super().__init__(env)

        self._t = 0
        self._window = window
        self._initialized = False
        self._plot_shown = False

        # set once the first reward_terms dict arrives
        self._term_names: list[str] = []
        self._fig: plt.Figure | None = None  # type: ignore[name-defined]
        self._axes_flat: list[plt.Axes] = []  # type: ignore[name-defined]
        self._lines: list = []
        self._line_total = None
        self._term_texts: list = []
        self._total_text = None

        self._ts: list[int] = []
        self._data: list[list[float]] = []
        self._data_total: list[float] = []

        plt.ion()

        def handler(sig, frame):
            plt.close("all")
            raise SystemExit(0)

        signal.signal(signal.SIGINT, handler)

    # ------------------------------------------------------------------
    # plot initialisation (deferred until first reward_terms is seen)
    # ------------------------------------------------------------------

    def _init_plots(self, term_names: list[str]) -> None:
        self._term_names = term_names
        n = len(term_names)
        n_total = n + 1  # extra slot for the total-reward subplot

        ncols = 2
        nrows = math.ceil(n_total / ncols)

        # allocate ~2.4 inches per row so there is room for text below each plot,
        # then clamp to fit within the screen
        fig_h = max(5.0, nrows * 2.4)
        self._fig, axes_2d = plt.subplots(
            nrows, ncols,
            sharex="all",
            squeeze=False,
            figsize=_fit_to_screen(9, fig_h),
        )
        # give room between rows for the annotation text
        self._fig.subplots_adjust(hspace=0.9, wspace=0.38, bottom=0.06, top=0.93)
        self._fig.suptitle("Reward Breakdown", fontsize=11)

        self._lines = []
        self._term_texts = []
        self._data = [[] for _ in range(n)]

        for i, name in enumerate(term_names):
            r, c = divmod(i, ncols)
            ax = axes_2d[r, c]
            self._axes_flat.append(ax)
            ax.set_title(name, fontsize=8, pad=2)
            ax.tick_params(labelsize=7)
            line, = ax.plot([], [], linewidth=1.1)
            self._lines.append(line)
            txt = ax.text(
                0.5, -0.38,
                "raw: —   scaled: —   |contrib|: —%",
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=7, color="#444444",
                clip_on=False,
            )
            self._term_texts.append(txt)

        # total-reward subplot
        r_tot, c_tot = divmod(n, ncols)
        ax_tot = axes_2d[r_tot, c_tot]
        self._axes_flat.append(ax_tot)
        ax_tot.set_title("total reward", fontsize=8, pad=2)
        ax_tot.tick_params(labelsize=7)
        self._line_total, = ax_tot.plot([], [], linewidth=1.1, color="black")
        self._total_text = ax_tot.text(
            0.5, -0.38,
            "total: —",
            transform=ax_tot.transAxes,
            ha="center", va="top",
            fontsize=7, color="#444444",
            clip_on=False,
        )

        # hide any unused slots
        for i in range(n_total, nrows * ncols):
            r, c = divmod(i, ncols)
            axes_2d[r, c].set_visible(False)

    # ------------------------------------------------------------------
    # per-step update
    # ------------------------------------------------------------------

    def _update_plots(self, reward: float, info: dict) -> None:
        terms: dict[str, float] = info.get("reward_terms", {})
        raw: dict[str, float] = info.get("reward_raw", {})

        if not terms:
            return

        if not self._initialized:
            self._init_plots(list(terms.keys()))
            self._initialized = True
            plt.show(block=False)
            plt.pause(0.01)
            self._plot_shown = True
            self._raise_pygame_window()

        self._ts.append(self._t)
        if len(self._ts) > self._window:
            self._ts = self._ts[-self._window:]
        ts = self._ts

        def _append(buf: list, val: float) -> None:
            buf.append(float(val))
            if len(buf) > self._window:
                del buf[0]

        for i, name in enumerate(self._term_names):
            _append(self._data[i], terms.get(name, 0.0))

        # windowed mean-absolute per term, then normalise across all terms so
        # sparse bonuses that rarely fire aren't swamped by their zero steps
        mean_abs = [
            sum(abs(v) for v in self._data[i]) / max(len(self._data[i]), 1)
            for i in range(len(self._term_names))
        ]
        total_mean_abs = sum(mean_abs)

        for i, name in enumerate(self._term_names):
            scaled = terms.get(name, 0.0)
            raw_val = raw.get(name, float("nan"))
            pct = (mean_abs[i] / total_mean_abs * 100.0) if total_mean_abs > 0 else 0.0

            self._lines[i].set_data(ts, self._data[i])
            ax = self._axes_flat[i]
            ax.relim()
            ax.autoscale_view()

            self._term_texts[i].set_text(
                f"raw: {raw_val:.3f}   scaled: {scaled:.3f}   |contrib|: {pct:.1f}%"
            )

        # total reward
        _append(self._data_total, reward)
        assert self._line_total is not None
        self._line_total.set_data(ts, self._data_total)
        self._axes_flat[-1].relim()
        self._axes_flat[-1].autoscale_view()
        assert self._total_text is not None
        self._total_text.set_text(f"total: {reward:.4f}")

        assert self._fig is not None
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    # ------------------------------------------------------------------
    # macOS window management
    # ------------------------------------------------------------------

    def _raise_pygame_window(self) -> None:
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

    # ------------------------------------------------------------------
    # gym interface
    # ------------------------------------------------------------------

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, term, trunc, info = super().step(action)
        self._t += 1
        self._update_plots(float(reward), info)
        return obs, reward, term, trunc, info
