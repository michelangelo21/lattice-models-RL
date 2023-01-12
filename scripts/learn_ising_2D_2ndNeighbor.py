# %%
from datetime import datetime

import gym
import numpy as np
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from src.custom_cnn import CustomCNN

# %%
SIDE_LENGTH = 4
env_id = "gym_xymodel:ising2d_2ndneighbor-v0"


def create_env(**env_kwargs):
    env = gym.make(env_id, **env_kwargs)
    env = TimeLimit(env, max_episode_steps=SIDE_LENGTH**2)
    # env = ContinuousLearningWrapper(env)
    return env


N_ENVS = 8

env = make_vec_env(
    create_env,
    env_kwargs=dict(L=SIDE_LENGTH, J1=-1.0, J2=-1.0),
    n_envs=N_ENVS,
    # wrapper_class=TimeLimit, # not here, as it is applied after the monitor wrapper
    # wrapper_kwargs=dict(max_episode_steps=SIDE_LENGTH**2),
)
env = VecMonitor(env)


# if Monitor is applied before TimeLimit, the env might run infinitely
eval_env = gym.make(env_id, L=SIDE_LENGTH, J1=-1.0, J2=-1.0)
eval_env = TimeLimit(eval_env, max_episode_steps=2 * SIDE_LENGTH**2)
eval_env = Monitor(eval_env)


# check_env(env)


# %%
# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=64),
#     net_arch=[{"pi": [64], "vf": [64]}],
# )

N_FEATURES = 128
policy_kwargs = dict(
    net_arch=[
        N_FEATURES,
        N_FEATURES,
        dict(vf=[N_FEATURES, N_FEATURES], pi=[N_FEATURES, N_FEATURES]),
    ]
)

date = datetime.now().strftime("%Y-%m-%dT%H%M%S")
# folder_path = f"../results/ising2D_2ndNN/L{SIDE_LENGTH}/{date}_3CNNcirc_feat64"
folder_path = f"../results/ising2D_2ndNN/L{SIDE_LENGTH}/{date}_mlp_nfeat{N_FEATURES}"

model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048 // N_ENVS,
    tensorboard_log=folder_path,
    verbose=0,
    policy_kwargs=policy_kwargs,
)

# model = PPO(
#     "CnnPolicy",
#     env,
#     policy_kwargs=policy_kwargs,
#     tensorboard_log=folder_path,
#     verbose=1,
# )

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=folder_path + "/logs/",
    log_path=folder_path + "/logs/",
    n_eval_episodes=20,
    eval_freq=max(5_000 // N_ENVS, 1),
    deterministic=True,
    render=False,
)

#%%
model.learn(500_000, callback=eval_callback)
model.save(f"{folder_path}/model")
