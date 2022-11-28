# %%
from datetime import datetime

import gym
import numpy as np
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

# from src.custom_cnn import CustomCNN
from src.custom_policy import CustomActorCriticPolicy, ReshapeExtractor

# %%

SIDE_LENGTH = 6
J = 1.0
D = 1.0
STEP_SIZE = 0.5
env = gym.make(
    "gym_xymodel:dzyaloshinskiimoriya2D-v0",
    L=SIDE_LENGTH,
    J=J,
    D=D,
    step_size=STEP_SIZE,
    max_episode_steps=4**2,
)
# env = TimeLimit(env, max_episode_steps=SIDE_LENGTH**2)

eval_env = gym.make(
    "gym_xymodel:dzyaloshinskiimoriya2D-v0",
    L=SIDE_LENGTH,
    J=J,
    D=D,
    step_size=STEP_SIZE,
    max_episode_steps=2 * 4**2,
)
# eval_env = TimeLimit(eval_env, max_episode_steps=2 * SIDE_LENGTH**2)


check_env(env)


# %%
# policy_kwargs = dict(
#     features_extractor_class=ReshapeExtractor,
#     net_arch={"n_filters": 64, "n_blocks": 2, "L": SIDE_LENGTH},
# )
n_features = 128
policy_kwargs = dict(
    net_arch=[
        n_features,
        n_features,
        dict(vf=[n_features, n_features], pi=[n_features, n_features]),
    ]
)

date = datetime.now().strftime("%Y-%m-%dT%H%M%S")
# folder_path = f"../results/xy2D/L{SIDE_LENGTH}/{date}_2CNNcirc_filters64"
folder_path = (
    f"../results/dzmoriya2D/L{SIDE_LENGTH}/{date}_J{J}_D{D}_step{STEP_SIZE}"
    + f"_mlp_nfeat{n_features}"
)

model = PPO(
    "MlpPolicy",
    env,
    tensorboard_log=folder_path,
    verbose=1,
    policy_kwargs=policy_kwargs,
)
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
    n_eval_episodes=10,
    eval_freq=10_000,
    deterministic=True,
    render=False,
)

# %%
model.learn(500_000, callback=eval_callback)
model.save(f"{folder_path}/model")

# %%
