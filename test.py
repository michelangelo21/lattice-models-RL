import gym
import numpy as np

env = gym.make("gym_xymodel:xymodel-v0")

env.reset()
env.step(env.action_space.sample())
print("yay")