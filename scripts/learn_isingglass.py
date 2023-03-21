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

# from src.custom_cnn import CustomCNN
from src.custom_policy import CustomActorCriticPolicy, ReshapeExtractor
from src.wrappers import ContinuousLearningWrapper

# %%

SIDE_LENGTH = 4
MAX_EPISODE_STEPS = 2 * SIDE_LENGTH**2
env_id = "gym_latticemodels:isingglass-v0"


def create_env(**env_kwargs):
    env = gym.make(env_id, **env_kwargs)
    # env = ContinuousLearningWrapper(env)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    return env


N_ENVS = 4

env = make_vec_env(
    create_env,
    env_kwargs=dict(L=SIDE_LENGTH),
    n_envs=N_ENVS,
    # wrapper_class=TimeLimit,
    # wrapper_kwargs=dict(max_episode_steps=SIDE_LENGTH**2),
)
env = VecMonitor(env)

eval_env = gym.make(env_id, L=SIDE_LENGTH)
eval_env = Monitor(TimeLimit(eval_env, max_episode_steps=MAX_EPISODE_STEPS))

# check_env(env)
N_FEATURES = 128
N_FILTERS = 64
N_BLOCKS = 3

# %%
# policy_kwargs = dict(
#     features_extractor_class=ReshapeExtractor,
#     net_arch={"n_filters": N_FILTERS, "n_blocks": N_BLOCKS, "L": SIDE_LENGTH},
# )

# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=64),
#     net_arch=[{"pi": [64], "vf": [64]}],
# )


policy_kwargs = dict(
    net_arch=[
        N_FEATURES,
        N_FEATURES,
        dict(vf=[N_FEATURES, N_FEATURES], pi=[N_FEATURES, N_FEATURES]),
    ]
)

date = datetime.now().strftime("%Y-%m-%dT%H%M%S")

folder_path = (
    f"../results/isingglass/L{SIDE_LENGTH}/{date}_mlp_steps_{MAX_EPISODE_STEPS}"
)
# folder_path = f"../results/ising2D/L{SIDE_LENGTH}/{date}_cstmpol_nfilters{N_FILTERS}_nblocks{N_BLOCKS}"

model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048 // N_ENVS,
    tensorboard_log=folder_path,
    verbose=0,
    # policy_kwargs=policy_kwargs,
)
# model = PPO(
#     CustomActorCriticPolicy,
#     env,
#     n_steps=2048 // N_ENVS,
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
    deterministic=False,
    render=False,
)

# %%
model.learn(1000_000, callback=eval_callback)
model.save(f"{folder_path}/model")
