"""
ppo_bc_sb3.common.dagger_buffer
===============================

storage for the dagger dataset D = {(s, a*)} where s is a state visited by the
current student policy and a* is the action the expert would take in that state.

how it gets populated (the dagger loop):

    1. during OnPolicyAlgorithm.collect_rollouts, the student picks an action and
       the env steps. for the same observation, query an expert model to get a*
       and call dagger_buffer.add(obs, a*).
    2. in PPO_BC._compute_policy_loss (or _compute_total_loss), sample a batch
       from dagger_buffer and add a bc term, e.g.
           bc_loss = MSE(policy.mean_actions(obs_batch), expert_actions_batch)
           total_loss = policy_loss + bc_coef * bc_loss + ...

storage is a simple preallocated ring buffer over numpy arrays. fast to add to,
samples are converted to torch on demand. swap to a different structure (e.g.
unbounded list, prioritized buffer) if needed for an experiment.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch as th

from stable_baselines3.common.utils import get_device


class DaggerBuffer:
    """
    fixed-size ring buffer holding (observation, expert_action) pairs.

    :param max_size: maximum number of samples retained. older entries are
        overwritten in fifo order once the buffer fills.
    :param obs_shape: shape of a single observation (without batch dim).
    :param action_shape: shape of a single action (without batch dim).
    :param device: torch device samples are moved to when sampled.
    :param obs_dtype: numpy dtype to store observations in.
    :param action_dtype: numpy dtype to store actions in.
    """

    def __init__(
        self,
        max_size: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: th.device | str = "auto",
        obs_dtype: np.dtype | type = np.float32,
        action_dtype: np.dtype | type = np.float32,
    ):
        assert max_size > 0, "max_size must be > 0"

        self.max_size = max_size
        self.obs_shape = tuple(obs_shape)
        self.action_shape = tuple(action_shape)
        self.device = get_device(device)

        # preallocate. we track pos (next write index) and full (whether we have
        # wrapped at least once) instead of a python list because batched env
        # steps push many samples per call and resizing is wasteful.
        self.observations = np.zeros((max_size, *self.obs_shape), dtype=obs_dtype)
        self.expert_actions = np.zeros((max_size, *self.action_shape), dtype=action_dtype)

        self.pos = 0
        self.full = False

    # ---- mutation ----------------------------------------------------------

    def add(self, obs: np.ndarray, expert_actions: np.ndarray) -> None:
        """
        append a batch of samples. obs and expert_actions must have matching
        leading batch dim. typically called from inside collect_rollouts after
        polling the expert on the same observation the student saw.
        """
        obs = np.asarray(obs)
        expert_actions = np.asarray(expert_actions)

        # promote single sample to batch of 1 so the rest of the path is uniform.
        if obs.shape == self.obs_shape:
            obs = obs[None, ...]
            expert_actions = expert_actions[None, ...]

        assert obs.shape[1:] == self.obs_shape, (
            f"obs shape {obs.shape[1:]} does not match expected {self.obs_shape}"
        )
        assert expert_actions.shape[1:] == self.action_shape, (
            f"action shape {expert_actions.shape[1:]} does not match expected {self.action_shape}"
        )
        assert obs.shape[0] == expert_actions.shape[0], (
            "obs and expert_actions must have matching batch dim"
        )

        n = obs.shape[0]

        # split the write across the ring boundary if needed.
        end = self.pos + n
        if end <= self.max_size:
            self.observations[self.pos:end] = obs
            self.expert_actions[self.pos:end] = expert_actions
            self.pos = end % self.max_size
            if end == self.max_size:
                self.full = True
        else:
            first = self.max_size - self.pos
            self.observations[self.pos:] = obs[:first]
            self.expert_actions[self.pos:] = expert_actions[:first]
            remaining = n - first
            self.observations[:remaining] = obs[first:]
            self.expert_actions[:remaining] = expert_actions[first:]
            self.pos = remaining
            self.full = True

    def clear(self) -> None:
        # discard all stored samples. does not deallocate the underlying arrays.
        self.pos = 0
        self.full = False

    # ---- access ------------------------------------------------------------

    def __len__(self) -> int:
        return self.max_size if self.full else self.pos

    def sample(self, batch_size: int) -> Tuple[th.Tensor, th.Tensor]:
        """
        uniformly sample a minibatch and return torch tensors on self.device.
        used inside the train loop to assemble the bc loss term.
        """
        assert len(self) > 0, "cannot sample from an empty DaggerBuffer"
        upper = len(self)
        idx = np.random.randint(0, upper, size=batch_size)
        obs = th.as_tensor(self.observations[idx], device=self.device)
        expert_actions = th.as_tensor(self.expert_actions[idx], device=self.device)
        return obs, expert_actions

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        # return the raw numpy slices currently in use. useful for offline
        # analysis or checkpointing the dataset.
        n = len(self)
        return self.observations[:n].copy(), self.expert_actions[:n].copy()
