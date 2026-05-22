"""
ppo_bc_sb3.common.on_policy_algorithm
=====================================

mirrors stable_baselines3.common.on_policy_algorithm.OnPolicyAlgorithm.

this is the base class that owns the outer training lifecycle:

    learn(total_timesteps):
        while num_timesteps < total_timesteps:
            collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
                rollout_buffer.reset()
                while n_steps < n_rollout_steps:
                    _predict_actions(self._last_obs)
                    _clip_actions(actions)
                    _step_envs(env, clipped_actions)
                    _handle_timeout_bootstrap(rewards, dones, infos)
                    _store_transition(rollout_buffer, ...)
                _compute_last_values_and_returns(rollout_buffer, new_obs, dones)
            train()        <- defined by PPO_BC subclass

we keep the public api of OnPolicyAlgorithm intact so anything that depended on
it in sb3 (callbacks, base_class hooks, save/load) keeps working. only the
internals of collect_rollouts have been split into smaller methods to make
DAgger expert polling easy to splice in: override _predict_actions or
_store_transition and the rest of the loop is untouched.

where to add expert polling (dagger):
    _predict_actions   <- query expert here on the same obs, push to DaggerBuffer
                          alongside the student's chosen action.

where to change rollout post-processing:
    _compute_last_values_and_returns  <- swap in v-trace, retrace, etc.
"""

from __future__ import annotations

import sys
import time
import warnings
from typing import Any, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

from ppo_bc_sb3.common.buffers import DictRolloutBuffer, RolloutBuffer

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for on-policy algorithms (e.g. A2C, PPO).

    Constructor and parameter semantics are identical to sb3's
    OnPolicyAlgorithm. See stable_baselines3.common.on_policy_algorithm for the
    full doc on each parameter.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule,
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        monitor_wrapper: bool = True,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: tuple[type[spaces.Space], ...] | None = None,
    ):
        # forward to sb3's BaseAlgorithm which handles env wrapping, seeds,
        # logger, callback plumbing, etc.
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        if _init_setup_model:
            self._setup_model()

    # ------------------------------------------------------------------
    # model setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        # called once at the end of __init__ (and again when loading from disk).
        # builds the lr schedule, the rollout buffer, and the policy network.
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # pick the right rollout buffer flavor for the obs space if the caller
        # did not specify one explicitly.
        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        # construct the policy. policy_class comes from BaseAlgorithm which
        # resolved the string alias (e.g. "MlpPolicy") to the actual class.
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
        self._maybe_recommend_cpu()

    def _maybe_recommend_cpu(self, mlp_class_name: str = "ActorCriticPolicy") -> None:
        # match sb3's hint that mlp policies usually want cpu, not gpu.
        policy_class_name = self.policy_class.__name__
        if self.device != th.device("cpu") and policy_class_name == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 "
                "for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU."
                "Note: The model will train, but the GPU utilization will be poor and "
                "the training might take longer than on CPU.",
                UserWarning,
            )

    # ------------------------------------------------------------------
    # rollout collection (the inner gym <-> policy loop)
    #
    # this is the place to splice in dagger expert polling. the cleanest entry
    # point is _predict_actions or _store_transition.
    # ------------------------------------------------------------------

    def _predict_actions(
        self, obs: np.ndarray | dict[str, np.ndarray]
    ) -> tuple[np.ndarray, th.Tensor, th.Tensor]:
        """
        run the policy forward to get actions, values, and log probs for the
        current observation. no gradient flow. returns numpy actions plus the
        torch tensors that the rollout buffer expects.

        override this to additionally poll an expert (for dagger) and stash
        (obs, expert_action) into a DaggerBuffer before returning.
        """
        with th.no_grad():
            obs_tensor = obs_as_tensor(obs, self.device)  # type: ignore[arg-type]
            actions, values, log_probs = self.policy(obs_tensor)
        return actions.cpu().numpy(), values, log_probs

    def _clip_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        rescale (if squash_output) or clip (otherwise) the policy's raw output
        so it lies inside the env action space. returns the numpy array that is
        actually fed to env.step. note the rollout buffer stores the
        unclipped, raw policy output (that's what gradients flow through).
        """
        clipped = actions
        if isinstance(self.action_space, spaces.Box):
            if self.policy.squash_output:
                clipped = self.policy.unscale_action(clipped)
            else:
                # unbounded gaussian samples can land outside [low, high]; clip.
                clipped = np.clip(actions, self.action_space.low, self.action_space.high)
        return clipped

    def _step_envs(
        self, env: VecEnv, clipped_actions: np.ndarray
    ) -> tuple[np.ndarray | dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """
        single env.step call. isolated so a custom training loop could insert
        timing, logging, or a different vec env api without touching the rest.
        """
        new_obs, rewards, dones, infos = env.step(clipped_actions)
        return new_obs, rewards, dones, infos

    def _handle_timeout_bootstrap(
        self, rewards: np.ndarray, dones: np.ndarray, infos: list[dict[str, Any]]
    ) -> np.ndarray:
        """
        for truncations (done due to timelimit, not actual terminal state) we
        add gamma * V(s_terminal) to the reward so the bootstrap is not lost.
        sb3 issue #633. mutates `rewards` in place but also returns it.
        """
        for idx, done in enumerate(dones):
            if (
                done
                and infos[idx].get("terminal_observation") is not None
                and infos[idx].get("TimeLimit.truncated", False)
            ):
                terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                with th.no_grad():
                    terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                rewards[idx] += self.gamma * terminal_value
        return rewards

    def _store_transition(
        self,
        rollout_buffer: RolloutBuffer,
        obs: np.ndarray | dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        episode_starts: np.ndarray,
        values: th.Tensor,
        log_probs: th.Tensor,
    ) -> None:
        """
        push one transition into the rollout buffer. thin shim around
        RolloutBuffer.add so subclasses can stash additional fields (e.g.
        expert actions stored timestep-aligned with rollouts).
        """
        rollout_buffer.add(obs, actions, rewards, episode_starts, values, log_probs)

    def _compute_last_values_and_returns(
        self,
        rollout_buffer: RolloutBuffer,
        new_obs: np.ndarray | dict[str, np.ndarray],
        dones: np.ndarray,
    ) -> None:
        """
        post-rollout step: bootstrap V(s_final) with the current critic and
        call compute_returns_and_advantage to fill in returns and gae
        advantages on the rollout buffer.
        """
        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill ``rollout_buffer``.
        Returns False if a callback asked to stop training early, True otherwise.

        the loop body is delegated to the small _* helpers above so dagger or
        any other modification has a clear seam to plug into.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # eval mode for batch norm / dropout. the optimization step in train()
        # flips it back to True.
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # gSDE: resample exploration weights at the start of every rollout.
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            # gSDE periodic resampling (different cadence from rollout reset).
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            # 1) ask the policy for an action on the current obs.
            #    DAGGER HOOK: override _predict_actions to also poll the expert
            #    on self._last_obs and add (obs, expert_a) to a DaggerBuffer.
            actions, values, log_probs = self._predict_actions(self._last_obs)

            # 2) prepare the action for env.step (clip / unscale).
            clipped_actions = self._clip_actions(actions)

            # 3) step the vec env.
            new_obs, rewards, dones, infos = self._step_envs(env, clipped_actions)

            self.num_timesteps += env.num_envs

            # 4) callback hooks. _on_step can stop training early by returning False.
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # match storage shape expected by rollout buffer for discrete actions.
                actions = actions.reshape(-1, 1)

            # 5) timelimit-truncation bootstrap. infos may carry
            #    terminal_observation + TimeLimit.truncated flags; if so, we
            #    bootstrap the reward with gamma * V(s_terminal).
            rewards = self._handle_timeout_bootstrap(rewards, dones, infos)

            # 6) store this transition into the rollout buffer.
            self._store_transition(
                rollout_buffer,
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        # rollout ended. fold the bootstrap value and run gae / return computation.
        self._compute_last_values_and_returns(rollout_buffer, new_obs, dones)

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    # ------------------------------------------------------------------
    # training step
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        Consume the current rollout buffer and update policy parameters.
        Implemented by individual algorithms (PPO_BC overrides this).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # logging + outer learn loop
    # ------------------------------------------------------------------

    def dump_logs(self, iteration: int = 0) -> None:
        # standard sb3 logging dump: fps, time elapsed, rollout reward stats.
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        # outer training lifecycle. collect_rollouts -> train -> repeat until
        # num_timesteps >= total_timesteps or a callback asks to stop.
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # periodic log dump (rollout stats from the just-collected batch).
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self.dump_logs(iteration)

            # train on the just-collected rollout buffer.
            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        # tells sb3's save/load machinery which attributes to (de)serialize.
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []
