# %%
from datetime import datetime

import gym
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gym.wrappers import TimeLimit
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# %%
side_length = 8
env = gym.make("gym_xymodel:isingmodel1dendrew-v0", L=side_length, max_steps=8)
env = TimeLimit(env, max_episode_steps=100)
from stable_baselines3.common.env_checker import check_env

check_env(env)
# env = gym.make('CartPole-v1')

# %%
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            #! code repetition
            n_flatten = self.cnn(
                F.pad(
                    th.as_tensor(observation_space.sample()[None])
                    .unsqueeze(-2)
                    .float(),
                    pad=(1, 1),
                    mode="circular",
                )
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # circular boundry conditions and add channel dimmension
        return self.linear(
            self.cnn(F.pad(observations.unsqueeze(-2), pad=(1, 1), mode="circular"))
        )


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

# %%
obs = env.observation_space.sample()
# obs.size
obs = torch.from_numpy(obs)
# m = nn.Conv1d(1, 6, 3, padding=1)
# # input = torch.randn(1, 1, 8)
# # # m(obs)
# # m(input)
# obs
# m(obs)
# F.pad(obs, pad=(1, 1), mode="circular")


# # %%

# from gym import spaces

# spaces.Box(low=0, high=1, shape=(1, 8))
# m(input).shape
# F.pad(input, (1, 1), mode="circular").shape
# input.shape


# %% training
date = datetime.now().strftime("%Y-%m-%dT%H%M%S")
folder_path = f"../results/ising1D_endreward/{date}_L{side_length}"
# model = PPO("MlpPolicy", env, tensorboard_log=folder_path)
model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    tensorboard_log=folder_path,
    verbose=True,
)
print(model.policy.features_extractor.cnn)
model.learn(total_timesteps=200_000)
model.save(f"{folder_path}/model")

# %%
energies = []
obs = env.reset()
for i in range(200):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    energies.append(env.compute_energy())
    if done:
        print(f"{reward=}")
        env.render()
        break
        # env.reset()
    # if i % 100 == 0:
    #   print(f"{i=}, {reward=}")
    #   env.render()


plt.plot(range(len(energies)), energies)

# %%
obs = env.reset()
env.render()
for i in range(10):
    action, _state = model.predict(obs, deterministic=True)
    print(f"{env.step_no=}")
    obs, reward, done, info = env.step(action)
    print(f"{i=}, {action=}, {done=}, {reward=}, {env.compute_energy()=}")
    env.render()

# %%
