import random
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class FalicovKimball1DEnv(gym.Env):
    """
    FalicovKimball1DEnv contains SIDE_LENGTH lattice
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, L: int = 8, max_steps: int = 4):
        """Initialization of the gym environment"""
        # lattice side_len
        self.L = L
        self.max_steps = max_steps
        self.step_no = 1

        self.observation_space = spaces.MultiBinary(self.L)
        # states are -1 or 1
        self.state = self.observation_space.sample()
        self.energy = self.compute_energy()
        self.action_space = spaces.Discrete(
            self.L + 1
        )  # +1 for pass action (end episode)

    # def state_to_lattice(self):
    #     """
    #     Convert state to lattice [0,1] -> [-1,1]
    #     """
    #     lattice = 2 * self.state - 1
    #     return lattice

    def compute_energy(self):
        """Computes energy of the current state per node (without optimalization yet"""
        # J=0 except for nearest neighbor
        lattice = self.state_to_lattice()
        energy = -sum(lattice[i] * (lattice[i - 1]) for i in range(self.L))
        return energy / self.L

    def step(self, action):
        info = {}

        # if action == self.L, skip round and do nothing,
        # this prevents agent from bouncing from the ground state
        # ? maybe done = True, and end round
        if action == self.L:
            reward = -self.compute_energy()
            done = True
            return self.state, float(reward), bool(done), info
        else:
            self.state[action] = 1 if self.state[action] == 0 else 0

        self.step_no += 1
        if self.step_no > self.max_steps:
            done = True
            reward = -self.compute_energy()
        else:
            done = False
            reward = 0

        return self.state, float(reward), bool(done), info

    def reset(self):
        self.state = self.observation_space.sample()
        self.step_no = 1
        return self.state

    def render(self, mode="human"):
        print(f"{self.state_to_lattice()}")

    # def close(self):
    #   ...
