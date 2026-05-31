"""
ppo_bc_sb3.ppo.ppo_bc
=====================

PPO clipped surrogate + behavior-cloning auxiliary loss with DAgger expert
relabeling. The hybrid objective per minibatch is

    L = L_PPO + bc_coef * L_BC + vf_coef * L_V + ent_coef * L_H

with L_BC = -E_{(s, a_E) ~ D}[ log pi_theta(a_E | s) ], where D is the
aggregated demo buffer built by OnPolicyAlgorithm via expert relabeling
(see arxiv 2212.11419 for the SAC-based original formulation).

load_expert() at the bottom is a thin SB3 PPO loader for use inside expert
callables passed to OnPolicyAlgorithm.
"""

from __future__ import annotations

import io
import pathlib
import warnings
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO as _SB3PPO
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance
from torch.nn import functional as F

from ppo_bc_sb3.common.buffers import RolloutBuffer
from ppo_bc_sb3.common.on_policy_algorithm import ExpertFn, OnPolicyAlgorithm
from wrappers.ppo_bc.ppo_bc_env import RlFTEnv

SelfPPO_BC = TypeVar("SelfPPO_BC", bound="PPO_BC")


class PPO_BC(OnPolicyAlgorithm):
    """PPO (clip) with a BC auxiliary loss fed by DAgger expert relabeling.

    See ``stable_baselines3.ppo.ppo.PPO`` for per-parameter docs of the inherited
    PPO args. DAgger / BC specific args:

    * ``experts``, ``task_bits``, ``act_var_floor``, ``dagger_max_size``:
      see ``OnPolicyAlgorithm``.
    * ``bc_coef``: weight on the BC loss term in the total loss.
    * ``bc_batch_size``: minibatch size sampled from the DAgger buffer per BC
      loss eval. Defaults to ``batch_size`` (the PPO minibatch size).
    * ``bc_loss_type``: ``"nll"`` for the negative-log-likelihood BC loss
      (default), or ``"mse"`` to regress the policy's deterministic action onto
      the expert action.
    """

    BC_LOSS_TYPES: ClassVar[tuple[str, ...]] = ("nll", "mse")

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        experts: dict[str, ExpertFn],
        task_bits: int,
        act_var_floor: float = 0.0,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        bc_coef: float = 0.1,
        bc_batch_size: int | None = None,
        bc_loss_type: str = "nll",
        collect_data: bool = True,
        adversarial_ag: bool = False,
        adversarial_eval_env: RlFTEnv | None = None,
        adversarial_eval_steps_per_task: int = 10000,
        adversarial_k: float = 0.85,
        dagger_max_size: int | None = None,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            experts=experts,
            task_bits=task_bits,
            act_var_floor=act_var_floor,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            bc_coef=bc_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            collect_data=collect_data,
            adversarial_ag=adversarial_ag,
            adversarial_eval_env=adversarial_eval_env,
            adversarial_eval_steps_per_task=adversarial_eval_steps_per_task,
            adversarial_k=adversarial_k,
            dagger_max_size=dagger_max_size,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be > 1 — see https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be > 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        # default BC minibatch to the PPO minibatch size unless overridden.
        self.bc_batch_size = bc_batch_size if bc_batch_size is not None else batch_size
        assert bc_loss_type in self.BC_LOSS_TYPES, (
            f"`bc_loss_type` must be one of {self.BC_LOSS_TYPES}, got {bc_loss_type!r}"
        )
        self.bc_loss_type = bc_loss_type

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        # promote float clip ranges to schedules so they can be evaluated on
        # current_progress_remaining each train() call.
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive; pass `None` to deactivate vf clipping"
                )
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)
            
    @classmethod
    def load(  # type: ignore[override]
        cls: type[SelfPPO_BC],
        path: str | pathlib.Path | io.BufferedIOBase,
        experts: dict[str, ExpertFn],
        task_bits: int,
        env: GymEnv | None = None,
        device: th.device | str = "auto",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfPPO_BC:
        """Inject experts + task_bits (not pickled) and defer to SB3's loader.

        BaseAlgorithm.load builds the model with ``cls(policy=..., env=..., device=..., _init_setup_model=False)``
        and ``PPO_BC.__init__`` requires experts/task_bits. We patch __init__
        for the duration of the super().load() call to supply them.
        """
        orig_init = cls.__init__

        def patched_init(self, *a, **kw):
            kw.setdefault("experts", experts)
            kw.setdefault("task_bits", task_bits)
            orig_init(self, *a, **kw)

        cls.__init__ = patched_init  # type: ignore[method-assign]
        try:
            model = super().load(
                path,
                env=env,
                device=device,
                custom_objects=custom_objects,
                print_system_info=print_system_info,
                force_reset=force_reset,
                **kwargs,
            )
        finally:
            cls.__init__ = orig_init  # type: ignore[method-assign]

        # super().load() does __dict__.update(data) which may stomp the
        # experts/task_bits we just injected (they're in _excluded_save_params,
        # but be defensive).
        model.experts = experts
        model.task_bits = task_bits
        return model

    # ---- loss + optimization helpers ---------------------------------------

    def _compute_advantages(self, rollout_data) -> th.Tensor:
        """
        pull the advantage tensor out of the minibatch, optionally normalize.
        normalization is skipped for batch size 1 since std() would be zero
        (sb3 GH issue #325).
        """
        advantages = rollout_data.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def _compute_ratio(self, log_prob: th.Tensor, old_log_prob: th.Tensor) -> th.Tensor:
        """
        importance sampling ratio between the current policy and the policy
        that collected the rollout. should be ~1 on the very first epoch.
        """
        return th.exp(log_prob - old_log_prob)

    def _compute_policy_loss(
        self, advantages: th.Tensor, ratio: th.Tensor, clip_range: float
    ) -> tuple[th.Tensor, float]:
        """Clipped surrogate objective; returns (loss, clip_fraction)."""
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
        with th.no_grad():
            clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
        return policy_loss, clip_fraction

    def _compute_bc_loss(self) -> th.Tensor:
        """BC loss on a DAgger-sampled minibatch (arxiv 2212.11419, adapted to PPO).

        ``bc_loss_type="nll"`` uses the negative log-likelihood of the expert
        action under the policy; ``"mse"`` regresses the policy's deterministic
        action (distribution mode) onto the expert action.

        Returns 0 when the buffer is empty (first rollout, before any
        aggregation).
        """
        if len(self.demo_dataset) == 0:
            return th.zeros((), device=self.device)

        obs, expert_act = self.demo_dataset.sample(self.bc_batch_size)
        # cast to the dtype the policy was built with (sb3 uses float32 by default).
        obs = obs.to(dtype=th.float32)
        expert_act = expert_act.to(dtype=th.float32)
        dist = self.policy.get_distribution(obs)
        if self.bc_loss_type == "mse":
            # dist.mode() is the deterministic action (gaussian mean, possibly
            # tanh-squashed) — regress it onto the expert action.
            return F.mse_loss(dist.mode(), expert_act)
        log_prob = dist.log_prob(expert_act)
        return -log_prob.mean()

    def _compute_value_loss(
        self,
        values: th.Tensor,
        old_values: th.Tensor,
        returns: th.Tensor,
        clip_range_vf: float | None,
    ) -> th.Tensor:
        if clip_range_vf is None:
            values_pred = values
        else:
            values_pred = old_values + th.clamp(
                values - old_values, -clip_range_vf, clip_range_vf
            )
        return F.mse_loss(returns, values_pred)

    def _compute_entropy_loss(
        self, entropy: th.Tensor | None, log_prob: th.Tensor
    ) -> th.Tensor:
        # fall back to -mean(log_prob) when entropy has no closed form (squashed gaussian).
        if entropy is None:
            return -th.mean(-log_prob)
        return -th.mean(entropy)

    def _compute_total_loss(
        self, policy_loss: th.Tensor, bc_loss: th.Tensor, value_loss: th.Tensor, entropy_loss: th.Tensor
    ) -> th.Tensor:
        return policy_loss + self.bc_coef * bc_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

    def _compute_approx_kl(self, log_prob: th.Tensor, old_log_prob: th.Tensor) -> np.ndarray:
        """
        schulman's k3 estimator for reverse kl. used to early-stop the epoch
        loop when the new policy drifts too far from the collecting one. see
        http://joschu.net/blog/kl-approx.html
        """
        with th.no_grad():
            log_ratio = log_prob - old_log_prob
            approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
        return approx_kl_div

    def _optimization_step(self, loss: th.Tensor) -> None:
        self.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

    # ------------------------------------------------------------------
    # outer training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """n_epochs passes over the rollout buffer, with KL early-stop."""

        # train mode flips batchnorm / dropout on. collect_rollouts flips it back.
        self.policy.set_training_mode(True)
        # apply the learning rate schedule.
        self._update_learning_rate(self.policy.optimizer)
        # current clip range for the policy. always a callable here because
        # _setup_model wrapped any float in a schedule.
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        clip_range_vf: float | None = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        # per-epoch diagnostic accumulators.
        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        bc_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []

        continue_training = True
        loss: th.Tensor | None = None

        for epoch in range(self.n_epochs):
            approx_kl_divs: list[np.ndarray] = []
            # iterate minibatches from the rollout buffer.
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # cast to long for discrete action spaces (categorical lik).
                    actions = rollout_data.actions.long().flatten()

                # forward pass for the current policy: V(s), log pi(a|s), H(pi).
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # 1) advantage normalization (per-minibatch).
                advantages = self._compute_advantages(rollout_data)

                # 2) importance sampling ratio between new and old policy.
                ratio = self._compute_ratio(log_prob, rollout_data.old_log_prob)

                # 3) clipped surrogate policy loss + clip fraction logging.
                policy_loss, clip_fraction = self._compute_policy_loss(advantages, ratio, clip_range)
                pg_losses.append(policy_loss.item())
                clip_fractions.append(clip_fraction)

                # 4) behavioral cloning loss
                bc_loss = self._compute_bc_loss()
                bc_losses.append(bc_loss.item())
                
                # 5) value loss (optionally clipped).
                value_loss = self._compute_value_loss(
                    values, rollout_data.old_values, rollout_data.returns, clip_range_vf
                )
                value_losses.append(value_loss.item())

                # 6) entropy bonus.
                entropy_loss = self._compute_entropy_loss(entropy, log_prob)
                entropy_losses.append(entropy_loss.item())

                # 7) compose the final scalar loss.
                loss = self._compute_total_loss(policy_loss, bc_loss, value_loss, entropy_loss)

                # 8) approximate kl, used both for logging and early stopping.
                approx_kl_div = self._compute_approx_kl(log_prob, rollout_data.old_log_prob)
                approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # 9) backward + optimizer step.
                self._optimization_step(loss)

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        if loss is not None:
            self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            # raw policy std (what the network learns) vs the effective std the
            # policy actually samples / scores with once act_var_floor is folded
            # in — the gap shows the floor holding off variance collapse.
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
            act_var_floor = getattr(self.policy, "_act_var_floor", 0.0)
            if act_var_floor > 0:
                eff_std = th.sqrt(th.exp(2.0 * self.policy.log_std) + act_var_floor)
                self.logger.record("train/std_effective", eff_std.mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfPPO_BC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO_BC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO_BC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


def load_expert(path, device: th.device | str = "cpu") -> _SB3PPO:
    """Load a frozen SB3 PPO expert in eval mode for use inside an expert callable."""
    model = _SB3PPO.load(path, device=device)
    model.policy.set_training_mode(False)
    return model
