# %%
from datetime import datetime
import copy

import gym
import numpy as np
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from src.custom_policy import CustomActorCriticPolicy, ReshapeExtractor

# %%

SIDE_LENGTH_0 = 4
env0 = gym.make("gym_latticemodels:ising2d-v0", L=SIDE_LENGTH_0, J=1.0)
env0 = TimeLimit(env0, max_episode_steps=SIDE_LENGTH_0**2)
model0 = PPO.load(
    "../results/ising2D/L4/2022-11-24T145201_1_2CNNcirc_filters64/model.zip", env=env0
)


SIDE_LENGTH = 4
env = gym.make("gym_latticemodels:ising2d-v0", L=SIDE_LENGTH, J=1.0)
env = TimeLimit(env, max_episode_steps=SIDE_LENGTH**2)

eval_env = gym.make("gym_latticemodels:ising2d-v0", L=SIDE_LENGTH, J=1.0)
eval_env = TimeLimit(eval_env, max_episode_steps=2 * SIDE_LENGTH**2)


check_env(env)


# %%
policy_kwargs = dict(
    features_extractor_class=ReshapeExtractor,
    net_arch={"n_filters": 64, "n_blocks": 2, "L": SIDE_LENGTH},
)

date = datetime.now().strftime("%Y-%m-%dT%H%M%S")
folder_path = (
    f"../results/ising2D/L{SIDE_LENGTH}/{date}_1_2CNNcirc_filters64_transfer_withgrad"
)

# model = PPO("MlpPolicy", env, tensorboard_log=folder_path, verbose=1)
model = PPO(
    CustomActorCriticPolicy,
    env,
    policy_kwargs=policy_kwargs,
    tensorboard_log=folder_path,
    verbose=1,
)

model.policy.mlp_extractor.shared_net = model0.policy.mlp_extractor.shared_net
model.policy.mlp_extractor.poicy_net = model0.policy.mlp_extractor.policy_net
model.policy.mlp_extractor.value_net = model0.policy.mlp_extractor.value_net
# model.policy.mlp_extractor = copy.deepcopy(model0.policy.mlp_extractor)
model.policy.mlp_extractor.requires_grad_(False)

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
model.learn(200_000, callback=eval_callback)
model.save(f"{folder_path}/model")
