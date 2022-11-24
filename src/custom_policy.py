from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from math import isqrt

import gym
import torch
from torch import nn

# from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device


def conv_block(
    in_channel: int,
    out_channel: int,
    kernel_size: int = 3,
    padding: int = 1,
    padding_mode: str = "circular",
    bias: bool = False,
) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias,
        ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    )


class ReshapeExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, 1)
        self.L = isqrt(observation_space.shape[0])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.reshape(observations, (-1, 1, self.L, self.L))


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Dict[str, List[int]],
        device: Union[torch.device, str] = "auto",
        # last_layer_dim_pi: int = 64,
        # last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        n_filters = net_arch["n_filters"]
        n_blocks = net_arch["n_blocks"]
        L = net_arch["L"]
        assert (
            L > 0 and L % 2 == 0
        ), """You must provide L in net_arch["L"], L must be even and positive."""

        assert feature_dim == 1

        device = get_device(device)

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = 2 * L**2
        # self.latent_dim_vf = last_layer_dim_vf
        self.latent_dim_vf = L**2

        # Shared network
        self.shared_net = nn.Sequential(
            conv_block(feature_dim, n_filters),
            *[conv_block(n_filters, n_filters) for _ in range(n_blocks)],
        ).to(device)

        # Policy network
        self.policy_net = nn.Sequential(
            conv_block(n_filters, 2, kernel_size=1, padding=0),
            nn.Flatten(),
        ).to(device)

        # Value network
        self.value_net = nn.Sequential(
            conv_block(n_filters, 1, kernel_size=1, padding=0),
            nn.Flatten(),
            # nn.Linear(L**2, last_layer_dim_vf),
            # nn.Tanh(),
        ).to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        x = self.shared_net(x)
        return self.policy_net(x), self.value_net(x)

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_net(self.shared_net(x))

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_net(self.shared_net(x))


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Dict[str, List[int]] = {"n_filters": 64, "n_blocks": 2, "L": None},
        # activation_fn: Type[nn.Module] = nn.Tanh,
        # features_extractor_class: Type[BaseFeaturesExtractor] = ReshapeExtractor,
        *args,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            # features_extractor_class=features_extractor_class
            # activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(
            self.features_dim,
            net_arch=self.net_arch,
            device=self.device,
        )
