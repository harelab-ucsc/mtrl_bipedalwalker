"""
ppo_bc_sb3.ppo.ppo_bc
=====================

mirrors stable_baselines3.ppo.ppo.PPO, renamed to PPO_BC, with train() split
into smaller methods so it is easy to add:

    - a behavior cloning (bc) loss term inside _compute_policy_loss or
      _compute_total_loss,
    - a dagger relabeling loop wired in via the OnPolicyAlgorithm hooks
      (see ppo_bc_sb3.common.on_policy_algorithm._predict_actions).

train() outer flow per call (one rollout has just been collected):

    set_training_mode(True)
    _update_learning_rate(optimizer)
    clip_range = self.clip_range(progress)
    for epoch in n_epochs:
        for batch in rollout_buffer.get(batch_size):
            advantages = _compute_advantages(batch)
            values, log_prob, entropy = policy.evaluate_actions(obs, actions)
            ratio = _compute_ratio(log_prob, batch.old_log_prob)
            policy_loss, clip_fraction = _compute_policy_loss(advantages, ratio, clip_range)
            value_loss = _compute_value_loss(values, batch.old_values, batch.returns, clip_range_vf)
            entropy_loss = _compute_entropy_loss(entropy, log_prob)
            loss = _compute_total_loss(policy_loss, value_loss, entropy_loss)
            approx_kl = _compute_approx_kl(log_prob, batch.old_log_prob)
            if early_stop_on_kl: break
            _optimization_step(loss)   # zero_grad + backward + clip + step
        self._n_updates += 1

at the bottom there's a thin load_expert() helper that returns an sb3 PPO model
ready to be polled inside the dagger hook on the collect side. it's just a
wrapper around stable_baselines3.PPO.load with eval mode set.
"""

from __future__ import annotations

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
from ppo_bc_sb3.common.on_policy_algorithm import OnPolicyAlgorithm

SelfPPO_BC = TypeVar("SelfPPO_BC", bound="PPO_BC")


class PPO_BC(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization (clip variant), prepared for adding a behavior
    cloning loss term and dagger relabeling. Constructor signature and defaults
    match stable_baselines3.ppo.PPO exactly so existing call sites translate
    one-for-one.

    See stable_baselines3.ppo.ppo.PPO for the per-parameter docs.
    """

    # default policy aliases. callers can also pass a class directly (we will
    # pass PpoBcPolicy from train.py).
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
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
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
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

        # advantage normalization breaks with batch size 1.
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # sanity check on n_steps * n_envs vs batch_size, same warning shape
            # as sb3 so user expectations carry over.
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
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

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # turn float clip ranges into schedules so we can interpolate over the
        # course of training. matches sb3 behavior.
        super()._setup_model()
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    # ------------------------------------------------------------------
    # loss + optimization helpers
    #
    # these are the methods you most likely want to edit to add the bc loss.
    # the natural insertion point is _compute_policy_loss (mix the bc term in
    # with the clipped surrogate) or _compute_total_loss (add as a separate
    # term with its own coefficient).
    # ------------------------------------------------------------------

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
        """
        clipped surrogate objective from the ppo paper. returns the scalar
        policy loss and the fraction of samples for which the ratio was
        outside the trust region (for logging).

        BC HOOK: this is where the behavior cloning term goes if you want it
        mixed directly into the actor's clipped surrogate. example:

            bc_obs, bc_expert_a = self.dagger_buffer.sample(self.bc_batch_size)
            student_mean = self.policy.get_distribution(bc_obs).distribution.mean
            bc_loss = F.mse_loss(student_mean, bc_expert_a)
            policy_loss = policy_loss + self.bc_coef * bc_loss
        """
        # unclipped objective.
        policy_loss_1 = advantages * ratio
        # clipped objective (limits the ratio to the trust region).
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        # ppo takes the minimum (pessimistic) of the two.
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        # fraction of the batch that hit the clip boundary, for diagnostics.
        with th.no_grad():
            clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()

        return policy_loss, clip_fraction

    def _compute_value_loss(
        self,
        values: th.Tensor,
        old_values: th.Tensor,
        returns: th.Tensor,
        clip_range_vf: float | None,
    ) -> th.Tensor:
        """
        value function loss: mse against the TD(lambda) returns target. if
        clip_range_vf is provided we additionally clip the value update to
        avoid jumps (openai-style).
        """
        if clip_range_vf is None:
            values_pred = values
        else:
            # constrain the value delta to ~ +/- clip_range_vf. note this
            # depends on the reward scale.
            values_pred = old_values + th.clamp(values - old_values, -clip_range_vf, clip_range_vf)
        return F.mse_loss(returns, values_pred)

    def _compute_entropy_loss(self, entropy: th.Tensor | None, log_prob: th.Tensor) -> th.Tensor:
        """
        entropy bonus to encourage exploration. when the distribution has no
        analytical entropy (e.g. squashed gaussian) we fall back to the negative
        log prob as a stochastic approximation.
        """
        if entropy is None:
            return -th.mean(-log_prob)
        return -th.mean(entropy)

    def _compute_total_loss(
        self, policy_loss: th.Tensor, value_loss: th.Tensor, entropy_loss: th.Tensor
    ) -> th.Tensor:
        """
        combine the three losses with their coefficients. add additional terms
        (e.g. a separate bc loss) here when wiring dagger.
        """
        return policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

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
        """
        single gradient step: zero, backward, clip grad norm, step.

        THIS IS THE .backward() CALL. when adding dagger or extra losses, you
        only need to make sure `loss` already includes them by the time you get
        here. global gradient clipping and the optimizer step itself stay the
        same.
        """
        self.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

    # ------------------------------------------------------------------
    # outer training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.

        runs n_epochs passes over the rollout, with optional early stopping when
        the approx kl exceeds 1.5 * target_kl. logging is unchanged from sb3.
        """
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

                # 4) value loss (optionally clipped).
                value_loss = self._compute_value_loss(
                    values, rollout_data.old_values, rollout_data.returns, clip_range_vf
                )
                value_losses.append(value_loss.item())

                # 5) entropy bonus.
                entropy_loss = self._compute_entropy_loss(entropy, log_prob)
                entropy_losses.append(entropy_loss.item())

                # 6) compose the final scalar loss.
                loss = self._compute_total_loss(policy_loss, value_loss, entropy_loss)

                # 7) approximate kl, used both for logging and early stopping.
                approx_kl_div = self._compute_approx_kl(log_prob, rollout_data.old_log_prob)
                approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # 8) backward + optimizer step.
                self._optimization_step(loss)

            self._n_updates += 1
            if not continue_training:
                break

        # explained variance over the entire rollout (after gae overwrote it).
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # diagnostic scalars for tensorboard.
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        if loss is not None:
            self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

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
        # forward to the base class learn() loop. nothing ppo-specific lives here.
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


# ----------------------------------------------------------------------
# expert loading helper
# ----------------------------------------------------------------------

def load_expert(path, device: th.device | str = "cpu") -> _SB3PPO:
    """
    load a frozen expert model from disk so it can be polled inside the dagger
    rollout hook. the returned object is a stable_baselines3.PPO instance with
    its policy in eval mode.

    the expert is purely a black box at inference time:

        action, _ = expert.predict(obs, deterministic=True)

    so we don't need our own PPO_BC for it. using sb3's PPO.load keeps things
    compatible with the existing checkpoints under models/experts/*.zip.

    :param path: path to the .zip file produced by PPO.save / CheckpointCallback.
    :param device: torch device the policy is loaded onto.
    """
    model = _SB3PPO.load(path, device=device)
    model.policy.set_training_mode(False)
    return model
