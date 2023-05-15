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
env_id = "gym_latticemodels:ladder2d-v0"
# env = gym.make(env_id, L=SIDE_LENGTH, max_episode_steps=4**2 + 10)
# env = ContinuousLearningWrapper(env)
# env = TimeLimit(env, max_episode_steps=4**2)


def create_env(**env_kwargs):
    env = gym.make(env_id, **env_kwargs)
    # env = ContinuousLearningWrapper(env)
    return env


N_ENVS = 8

env = make_vec_env(
    create_env,
    env_kwargs=dict(L=SIDE_LENGTH, max_episode_steps=MAX_EPISODE_STEPS),
    n_envs=N_ENVS,
    # wrapper_class=TimeLimit,
    # wrapper_kwargs=dict(max_episode_steps=4**2),
)
env = VecMonitor(env)

eval_env = Monitor(
    gym.make(env_id, L=SIDE_LENGTH, max_episode_steps=2 * MAX_EPISODE_STEPS)
)
# eval_env = Monitor(TimeLimit(eval_env, max_episode_steps=2 * 4**2))

# check_env(env)

# %%


date = datetime.now().strftime("%Y-%m-%dT%H%M%S")
# folder_path = f"../results/xy2D/L{SIDE_LENGTH}/{date}_2CNNcirc_filters64"
# folder_path = f"../results/xy2D/L{SIDE_LENGTH}/{date}_mlp_continuous_nenvs{N_ENVS}"
# folder_path = (
#     f"../results/xy2D/L{SIDE_LENGTH}/{date}_mlp_nenvs{N_ENVS}_nsteps{N_STEPS}"
# )

N_FEATURES = 256
folder_path = f"../results/ladder/L{SIDE_LENGTH}/{date}_mlp_nenvs{N_ENVS}_nfeatures{N_FEATURES}_nblocks3"
# folder_path = f"../results/xy2D/L{SIDE_LENGTH}/{date}_cnn_nenvs{N_ENVS}_nfeatures{N_FEATURES}_nblocks2"


policy_kwargs = dict(
    net_arch=[
        dict(
            pi=[N_FEATURES, N_FEATURES, N_FEATURES],
            vf=[N_FEATURES, N_FEATURES, N_FEATURES],
        )
    ]
)
model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048 // N_ENVS,
    tensorboard_log=folder_path,
    verbose=1,
)

# policy_kwargs = dict(
#     features_extractor_class=ReshapeExtractor,
#     net_arch={"n_filters": N_FEATURES, "n_blocks": 2, "L": SIDE_LENGTH},
# )
# model = PPO(
#     CustomActorCriticPolicy,
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

# %%
model.learn(1000_000, callback=eval_callback)
model.save(f"{folder_path}/model")
