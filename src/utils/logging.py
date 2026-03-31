import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class StandardTBCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._rollout_ep_rewards = []

    def _on_rollout_start(self) -> None:
        self._rollout_ep_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._rollout_ep_rewards.append(info["episode"]["r"])
        return True

    def _on_rollout_end(self) -> None:    
        if self._rollout_ep_rewards:
            self.logger.record("rollout/ep_rew_min_raw", np.min(self._rollout_ep_rewards))
            self.logger.record("rollout/ep_rew_mean_raw", np.mean(self._rollout_ep_rewards))
            self.logger.record("rollout/ep_rew_max_raw", np.max(self._rollout_ep_rewards))
        
        buf = self.model.ep_info_buffer
        if buf:
            rew = [ep["r"] for ep in buf]
            self.logger.record("rollout/ep_rew_min", np.min(rew))
            self.logger.record("rollout/ep_rew_max", np.max(rew))


class RewardTermLogger(BaseCallback):
    """
    Logs per-step reward component values stored in info["reward_terms"].

    Any wrapper can populate info["reward_terms"] = {"term_name": value, ...}
    and this callback will record the rollout mean for each term to TensorBoard
    under reward_terms/{term_name}.

    Args:
        keys: Which component keys to log. None logs all keys present in infos.
    """

    def __init__(self, keys: list[str] | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.keys = keys
        self._accum: dict[str, list[float]] = {}

    def _on_rollout_start(self) -> None:
        self._accum = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            components: dict = info.get("reward_terms", {})
            items = components.items() if self.keys is None else (
                (k, components[k]) for k in self.keys if k in components
            )
            for k, v in items:
                self._accum.setdefault(k, []).append(v)
        return True

    def _on_rollout_end(self) -> None:
        for k, vals in self._accum.items():
            self.logger.record(f"reward_terms/{k}", np.mean(vals))


def fmt_duration(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h: parts.append(f"{int(h)}h")
    if m: parts.append(f"{int(m)}m")
    parts.append(f"{s:.2f}s")
    return " ".join(parts)