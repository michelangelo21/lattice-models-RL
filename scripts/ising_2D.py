# %%
from datetime import datetime

import gym
import numpy as np
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from src.custom_cnn import CustomCNN

# %%

SIDE_LENGTH = 6
env = gym.make("gym_xymodel:ising2d-v0", L=SIDE_LENGTH, J=1.0)
env = TimeLimit(env, max_episode_steps=SIDE_LENGTH**2)

check_env(env)

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)


# %%
date = datetime.now().strftime("%Y-%m-%dT%H%M%S")
folder_path = f"../results/ising2D/{date}_L{SIDE_LENGTH}_CNN"

# model = PPO("MlpPolicy", env, tensorboard_log=folder_path, verbose=1)
model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    tensorboard_log=folder_path,
    verbose=1,
)
model.learn(200_000)
model.save(f"{folder_path}/model")
