from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch.nn.functional as F


class CustomCNN1D(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN1D, self).__init__(observation_space, features_dim)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.cnn = nn.Sequential(  # 1 input channel
            nn.Conv1d(1, 8, kernel_size=3, padding=0),  # 8 possible combinations
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            #! code repetition
            n_flatten = self.cnn(
                F.pad(
                    torch.as_tensor(observation_space.sample()[None])
                    .unsqueeze(-2)
                    .float(),
                    pad=(1, 1),
                    mode="circular",
                )
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # circular boundry conditions and add channel dimmension
        return self.linear(
            self.cnn(F.pad(observations.unsqueeze(-2), pad=(1, 1), mode="circular"))
        )


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)
