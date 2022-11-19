from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # Re-ordering will be done by pre-preprocessing or wrapper
        conv_features_dim = 32
        self.L = int(np.sqrt(observation_space.shape[0]))
        # TODO refactor to ConvBlock
        self.cnn = nn.Sequential(  # 1 input channel
            nn.Conv2d(
                1, conv_features_dim, kernel_size=3, padding=1, padding_mode="zeros"
            ),  # 8 possible combinations
            nn.BatchNorm2d(conv_features_dim),
            nn.ReLU(),
            nn.Conv2d(
                conv_features_dim,
                conv_features_dim,
                kernel_size=3,
                padding=1,
                padding_mode="zeros",
            ),
            nn.BatchNorm2d(conv_features_dim),
            nn.ReLU(),
            nn.Conv2d(
                conv_features_dim, 3, kernel_size=3, padding=1, padding_mode="zeros"
            ),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            #! code repetition
            s = torch.as_tensor(observation_space.sample()[None]).float()
            s = torch.reshape(
                s,
                (-1, 1, self.L, self.L),
            )
            n_flatten = self.cnn(s).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # circular boundry conditions and add channel dimmension
        return self.linear(
            self.cnn(torch.reshape(observations, (-1, 1, self.L, self.L)))
        )
