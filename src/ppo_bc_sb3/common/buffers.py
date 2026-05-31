"""
ppo_bc_sb3.common.buffers
=========================

mirrors stable_baselines3.common.buffers. only the on-policy bits are kept here:

    BaseBuffer        -> shared base for all buffer types
    RolloutBuffer     -> the on-policy buffer used by PPO / A2C
    DictRolloutBuffer -> dict-obs variant of RolloutBuffer

the off-policy buffers (ReplayBuffer, DictReplayBuffer, NStepReplayBuffer) live in
sb3 itself but are not used by PPO, so they are not duplicated here. if you ever
need them, import directly from stable_baselines3.

this file is intentionally close to a verbatim copy of the sb3 source so that:

  1. behavior stays identical when nothing has been changed yet,
  2. the rollout post-processing (compute_returns_and_advantage, get / _get_samples)
     is easy to find and modify, e.g. when stashing extra fields like expert actions.

flow:

    OnPolicyAlgorithm.collect_rollouts
        rollout_buffer.reset()
        for step in n_rollout_steps:
            rollout_buffer.add(obs, action, reward, episode_start, value, log_prob)
        rollout_buffer.compute_returns_and_advantage(last_values, dones)

    PPO_BC.train
        for epoch in n_epochs:
            for batch in rollout_buffer.get(batch_size):
                ... compute losses, backward ...
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictRolloutBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay).

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        # collapse [n_steps, n_envs, ...] into [n_steps * n_envs, ...] preserving order.
        # used right before yielding minibatches in get() so the train loop sees a flat
        # batch dimension instead of a (steps, envs) grid.
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        for data in zip(*args, strict=True):
            self.add(*data)

    def reset(self) -> None:
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: VecNormalize | None = None):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None):
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(
        obs: np.ndarray | dict[str, np.ndarray],
        env: VecNormalize | None = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: VecNormalize | None = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO. Holds ``buffer_size``
    transitions collected with the current policy. Discarded after each policy update.

    stored fields per step (shape [buffer_size, n_envs, ...]):

        observations    -> the observation that produced the action
        actions         -> the action taken
        rewards         -> reward received for that action
        episode_starts  -> 1.0 if this step was the start of an episode
        values          -> V(s) from the critic when the action was taken
        log_probs       -> log pi(a | s) at the time the action was taken
        advantages      -> filled in by compute_returns_and_advantage
        returns         -> filled in by compute_returns_and_advantage

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for GAE
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        # pre-allocate the storage arrays. called once on construction and once at the
        # start of every rollout collection pass.
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self.observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        post-processing step run at the end of every rollout. computes the
        TD(lambda) return target and the GAE(lambda) advantage in one backward pass.

        modify this method if you want a different return / advantage estimator
        (e.g. monte carlo only, v-trace, retrace).
        """
        # convert the bootstrap value tensor to a numpy array on cpu so the
        # downstream loop is pure numpy.
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        # walk the rollout backwards in time to accumulate the gae advantage.
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) target = advantage + value baseline.
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        push one (vectorized across n_envs) transition into the buffer. called once
        per env step from OnPolicyAlgorithm._store_transition.
        """
        if len(log_prob.shape) == 0:
            # reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: int | None = None) -> Generator[RolloutBufferSamples, None, None]:
        """
        yield minibatches of rollout data. called from PPO_BC.train once the buffer
        is full. the first call flattens the (steps, envs) grid into one batch axis.
        """
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor]) # type: ignore
            self.generator_ready = True

        # return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> RolloutBufferSamples:
        # builds a RolloutBufferSamples namedtuple ready for the loss computation.
        # cast actions to float32 for backward compatibility (MultiBinary stays as-is).
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class DictRolloutBuffer(RolloutBuffer):
    """
    Dict-observation variant of RolloutBuffer. Used automatically by
    OnPolicyAlgorithm when the env exposes a spaces.Dict observation space.

    flow is identical to RolloutBuffer, only the storage shape changes:
    observations is a dict[str, np.ndarray] keyed by space key.
    """

    observation_space: spaces.Dict
    obs_shape: dict[str, tuple[int, ...]]  # type: ignore[assignment]
    observations: dict[str, np.ndarray]  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        # skip the RolloutBuffer.__init__ allocation since we re-allocate as dicts.
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros(
                (self.buffer_size, self.n_envs, *obs_input_shape), dtype=self.observation_space[key].dtype
            )
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key])
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(  # type: ignore[override]
        self,
        batch_size: int | None = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor]) # type: ignore
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> DictRolloutBufferSamples:
        return DictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds].astype(np.float32, copy=False)),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )
