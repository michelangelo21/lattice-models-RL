# %%
from datetime import datetime

import gym
import numpy as np
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from src.custom_cnn import CustomCNN

# %%

SIDE_LENGTH = 4
env = gym.make("gym_xymodel:ising2d-v0", L=SIDE_LENGTH, J=1.0)
env = TimeLimit(env, max_episode_steps=SIDE_LENGTH**2)

eval_env = gym.make("gym_xymodel:ising2d-v0", L=SIDE_LENGTH, J=1.0)
eval_env = TimeLimit(eval_env, max_episode_steps=2 * SIDE_LENGTH**2)


check_env(env)


# %%
date = datetime.now().strftime("%Y-%m-%dT%H%M%S")
folder_path = f"../results/ising2D/L{SIDE_LENGTH}/{date}_2CNN_feat32"

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=32),
    net_arch=[{"pi": [32], "vf": [32]}],
)

# model = PPO("MlpPolicy", env, tensorboard_log=folder_path, verbose=1)
model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    tensorboard_log=folder_path,
    verbose=1,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=folder_path + "/logs/",
    log_path=folder_path + "/logs/",
    eval_freq=2000,
    deterministic=True,
    render=False,
)

#%%
model.learn(200_000, callback=eval_callback)
model.save(f"{folder_path}/model")
