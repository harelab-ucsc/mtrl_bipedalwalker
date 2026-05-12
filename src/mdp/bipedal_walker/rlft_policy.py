from typing import Callable

import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy


_MODEL_CONFIGS = {
    "xs":   [128, 64, 32],
    "s":    [192, 96, 48],
    "m":    [256, 128, 64],
    "ml":   [320, 160, 80],
    "l":    [384, 192, 96],
    "xl":   [512, 256, 128],
    "xll":  [768, 384, 192],
    "xlll": [1024, 512, 256],
    "xxxl": [1024, 512, 512, 256, 256],
}


class RlFTNetwork(nn.Module):
    def __init__(self, feature_dim: int, hidden_dims: list, critic_hidden_dims: list, activation: type):
        super().__init__()

        def _make_net(in_dim, hidden):
            layers, d = [], in_dim
            for h in hidden:
                layers += [nn.Linear(d, h), activation()]
                d = h
            return nn.Sequential(*layers), d

        self.policy_net, self.latent_dim_pi = _make_net(feature_dim, hidden_dims)
        self.value_net, self.latent_dim_vf = _make_net(feature_dim, critic_hidden_dims)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class RlFTPolicy(ActorCriticPolicy):
    def __init__(
        self,
        obs_space: spaces.Space,
        act_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        hidden_dims,
        critic_hidden_dims=None,
        **kwargs,
    ):
        self._hidden_dims = hidden_dims
        self._critic_hidden_dims = critic_hidden_dims if critic_hidden_dims is not None else _MODEL_CONFIGS["xl"]
        kwargs["ortho_init"] = False
        super().__init__(obs_space, act_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = RlFTNetwork(
            self.features_dim,
            self._hidden_dims,
            self._critic_hidden_dims,
            self.activation_fn,
        )
