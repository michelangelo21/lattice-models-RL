import gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResBlock(nn.Module):
    def __init__(self, in_out_size: int, hidden_size: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_out_size),
            nn.ReLU(),
            nn.Conv2d(
                in_out_size,
                hidden_size,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
                bias=False,
            ),
        )
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Conv2d(
                hidden_size,
                in_out_size,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
                bias=False,
            ),
        )

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x


class CustomResNet2D(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.Space, features_dim: int = 64, n_blocks: int = 2
    ):
        super().__init__(observation_space, features_dim)
        self.cnn0 = nn.Conv2d(
            1,
            features_dim,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        self.resnet = nn.Sequential(
            *[ResBlock(features_dim, features_dim) for _ in range(n_blocks)],
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.resnet(self.cnn0(observations))
