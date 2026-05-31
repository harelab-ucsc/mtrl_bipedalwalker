"""
ppo_bc_sb3.common.policies
==========================

actor / critic policy and network for our ppo+bc work.

we deliberately subclass sb3's ActorCriticPolicy rather than copying its 600+
line body, since most of the surface (action distribution selection, gSDE
plumbing, save/load, predict) is fine as-is and re-implementing it would just
make us drift from the upstream api. what we DO override is:

    PpoBcPolicy._build_mlp_extractor
        plugs in our PpoBcNetwork instead of sb3's MlpExtractor. PpoBcNetwork
        is a near-clone of src/mdp/bipedal_walker/rlft_policy.py:RlFTNetwork
        so the layout is familiar.

the forward path looks like this (inherited from sb3, kept here for reference):

    obs
      -> features_extractor (FlattenExtractor by default for vector obs)
      -> mlp_extractor.forward_actor      (PpoBcNetwork.policy_net)
                                          gives latent_pi
      -> action_net (linear) + log_std    (DiagGaussianDistribution by default)
                                          gives action distribution
      -> sample / log_prob                gives action, log_prob
      ALSO:
      -> mlp_extractor.forward_critic     (PpoBcNetwork.value_net)
                                          gives latent_vf
      -> value_net (linear)               gives V(s)

customization hooks:

    - architecture: swap PpoBcNetwork or override _build_mlp_extractor.
    - optimizer:    pass optimizer_class / optimizer_kwargs through policy_kwargs.
                    they reach this class via ActorCriticPolicy.__init__.
    - log_std init: pass log_std_init through policy_kwargs.
    - ortho_init:   default off (matches RlFTPolicy). flip via policy_kwargs.
"""

from __future__ import annotations

from typing import Any, Callable

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class PpoBcNetwork(nn.Module):
    """
    actor + critic mlp extractor with independent trunks.

    structure (per trunk):
        Linear(in_dim, hidden[0]) -> activation
        Linear(hidden[0], hidden[1]) -> activation
        ...
        (output of the last hidden layer becomes the latent for the head)

    the policy head (action_net) is appended by ActorCriticPolicy._build after
    this, so latent_dim_pi / latent_dim_vf must match the last hidden dim.

    :param feature_dim: number of features coming out of the features_extractor.
    :param hidden_dims: list of hidden layer widths for the actor trunk.
    :param critic_hidden_dims: list of hidden layer widths for the critic trunk.
    :param activation: activation class, instantiated with no args at each layer.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: type[nn.Module],
    ):
        super().__init__()

        # helper to build a simple Sequential mlp. extracted so the actor and
        # critic share construction logic; replace this if you want layernorm,
        # residual blocks, or different activations per layer.
        def _make_net(in_dim: int, hidden: list[int]) -> tuple[nn.Sequential, int]:
            layers: list[nn.Module] = []
            d = in_dim
            for h in hidden:
                layers += [nn.Linear(d, h), activation()]
                d = h
            return nn.Sequential(*layers), d

        self.policy_net, self.latent_dim_pi = _make_net(feature_dim, hidden_dims)
        self.value_net, self.latent_dim_vf = _make_net(feature_dim, critic_hidden_dims)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        # sb3 expects (latent_pi, latent_vf) from the shared mlp_extractor path.
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class PpoBcPolicy(ActorCriticPolicy):
    """
    actor-critic policy with explicit actor / critic widths and orthogonal
    initialization disabled by default (matches the rlft policy convention).

    constructor extras vs ActorCriticPolicy:
        hidden_dims         -> actor trunk hidden widths
        critic_hidden_dims  -> critic trunk hidden widths (defaults to hidden_dims)
        act_var_floor       -> additive variance bias / floor on the (diagonal
                               gaussian) action distribution. See
                               _get_action_dist_from_latent.

    everything else (log_std_init, use_sde, optimizer_class, optimizer_kwargs,
    features_extractor_class, ...) is passed through **kwargs to the sb3 base.

    typical usage:

        model = PPO_BC(
            PpoBcPolicy,
            env=train_env,
            policy_kwargs=dict(
                hidden_dims=[256, 128, 64],
                critic_hidden_dims=[256, 128, 64],
                activation_fn=torch.nn.ELU,
                log_std_init=0.0,
            ),
            ...
        )
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        hidden_dims: list[int],
        critic_hidden_dims: list[int] | None = None,
        act_var_floor: float = 0.0,
        **kwargs: Any,
    ):
        # store on self before super().__init__ because ActorCriticPolicy._build
        # runs from inside super().__init__ and calls _build_mlp_extractor which
        # needs these attributes to be set.
        self._hidden_dims = list(hidden_dims)
        self._critic_hidden_dims = (
            list(critic_hidden_dims) if critic_hidden_dims is not None else list(hidden_dims)
        )
        # additive variance bias on the student's action distribution; see
        # _get_action_dist_from_latent. 0.0 disables it (plain ActorCriticPolicy
        # behavior). Consumed here, NOT forwarded to the sb3 base.
        self._act_var_floor = float(act_var_floor)

        # default to ortho_init=False to match RlFTPolicy. caller can still flip
        # it via policy_kwargs since kwargs is forwarded to ActorCriticPolicy.
        kwargs.setdefault("ortho_init", False)

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        # called from ActorCriticPolicy._build during __init__. wire in our own
        # actor / critic trunks; the rest of the build (action_net, value_net,
        # optimizer) continues in the base class with the latent dims we expose.
        self.mlp_extractor = PpoBcNetwork(
            feature_dim=self.features_dim,
            hidden_dims=self._hidden_dims,
            critic_hidden_dims=self._critic_hidden_dims,
            activation=self.activation_fn,
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        """Retrieve action distribution given the latent codes.

        Extra vs sb3: for the diagonal gaussian head we bias the variance by
        ``act_var_floor`` (var_eff = exp(2*log_std) + act_var_floor) so the
        policy can't collapse to a deterministic action under the BC loss. This
        is the single choke point for sb3's forward / evaluate_actions /
        get_distribution, so the floor applies uniformly to sampling, the PPO
        ratio + entropy, and the BC loss. Disabled (<= 0) or non-gaussian heads
        defer to the base implementation.
        """
        if self._act_var_floor > 0 and isinstance(
            self.action_dist, DiagGaussianDistribution
        ):
            mean_actions = self.action_net(latent_pi)
            log_std_eff = 0.5 * th.log(
                th.exp(2.0 * self.log_std) + self._act_var_floor
            )
            return self.action_dist.proba_distribution(mean_actions, log_std_eff)
        return super()._get_action_dist_from_latent(latent_pi)
