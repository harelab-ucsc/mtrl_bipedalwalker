from __future__ import annotations

import os
from typing import Tuple, cast

import numpy as np
import torch as th

from stable_baselines3.common.utils import get_device

from mdp.bipedal_walker.tasks import GAIT


class DaggerDataset:
    """(obs, expert_action) pairs aggregated across DAgger iterations."""

    def __init__(
        self,
        device: th.device | str = "auto",
        max_size: int | None = None,
    ):
        self.device = get_device(device)
        # None disables the cap entirely (unbounded growth).
        self.max_size = max_size
        self.D: list[tuple[np.ndarray, np.ndarray]] = []

    # ---- mutation ----------------------------------------------------------

    def add(self, obs: np.ndarray, act: np.ndarray) -> None:
        self.D.append((obs, act))
        self._maybe_evict()

    def extend(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.D.extend(pairs)
        self._maybe_evict()

    def clear(self) -> None:
        self.D = []

    def _maybe_evict(self) -> None:
        """
        Recency-biased eviction: when the buffer overshoots max_size, keep
        max_size entries sampled without replacement with weight w_i = i+1
        (oldest = lowest keep-prob, newest = highest). Older samples are
        preferentially dropped but very recent ones can still be evicted, which
        preserves some early-distribution coverage instead of collapsing to a
        pure FIFO tail.
        """
        if self.max_size is None or len(self.D) <= self.max_size:
            return
        n = len(self.D)
        # Linearly increasing weights — newest gets the highest keep probability.
        w = np.arange(1, n + 1, dtype=np.float64)
        w /= w.sum()
        keep = np.random.choice(n, size=self.max_size, replace=False, p=w)
        keep.sort()  # preserve insertion order so weights stay meaningful next time
        self.D = [self.D[i] for i in keep]

    # ---- access ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.D)

    def sample(self, batch_size: int) -> Tuple[th.Tensor, th.Tensor]:
        """Uniform minibatch as torch tensors on self.device."""
        assert len(self) > 0, "cannot sample from an empty DaggerDataset"
        idx = np.random.randint(0, len(self.D), size=batch_size)
        obs = np.stack([self.D[i][0] for i in idx])
        expert_actions = np.stack([self.D[i][1] for i in idx])
        return (
            th.as_tensor(obs, device=self.device),
            th.as_tensor(expert_actions, device=self.device),
        )

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.stack([s for s, _ in self.D])
        expert_actions = np.stack([a for _, a in self.D])
        return obs, expert_actions

    def task_counts(self, task_bits: int, scheme: str = GAIT) -> dict[str, int]:
        """Count entries by directional task name, derived from each stored obs's
        task bits + commands (scheme-aware; see tasks.resolve_task). Combined /
        unrecognized vectors bucket under their composed name. One pass over D.
        Layout: [..., cmd_vel, cmd_tilt, *task_bits], so cmd_vel = obs[-task_bits-2]
        and cmd_tilt = obs[-task_bits-1]."""
        from mdp.bipedal_walker.tasks import _name_from_bits, resolve_single_task

        counts: dict[str, int] = {}
        for obs, _ in self.D:
            bits = cast(tuple[int, int, int], (int(x) for x in obs[-task_bits:]))
            spec = resolve_single_task(
                bits, float(obs[-task_bits - 2]), float(obs[-task_bits - 1]), scheme
            )
            key = spec.name if spec is not None else _name_from_bits(bits)
            counts[key] = counts.get(key, 0) + 1
        return counts
    
    # ---- save / load -------------------------------------------------------

    def dump(self, path: str) -> None:
        """Save aggregated (obs, expert_action) pairs as a compressed .npz.

        File layout: arrays ``obs`` [N, obs_dim] and ``expert_actions`` [N, act_dim].
        No-op if the dataset is empty.
        """
        if len(self.D) == 0:
            return
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        obs, expert_actions = self.get_all()
    
        np.savez_compressed(path, obs=obs, expert_actions=expert_actions)

    @classmethod
    def load(
        cls,
        path: str,
        device: th.device | str = "auto",
        max_size: int | None = None,
    ) -> "DaggerDataset":
        """Reconstruct a DaggerDataset from a .npz produced by ``dump``."""
        data = np.load(path)
        obs, expert_actions = data["obs"], data["expert_actions"]
        ds = cls(device=device, max_size=max_size)
        ds.D = [(obs[i], expert_actions[i]) for i in range(len(obs))]
        return ds
