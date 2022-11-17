import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class Ising2DEnv(gym.Env):
    """
    Ising2DEnv contains SIDE_LENGTH x SIDE_LENGTH lattice
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, L: int = 4, J: float = 1.0):
        """Initialization of the gym environment"""
        # lattice side_len x side_len
        self.L = L
        self.J = J

        self.observation_space = spaces.MultiBinary(self.L**2)
        # states are -1 or 1
        self.state = self.observation_space.sample()
        self.action_space = spaces.Discrete(self.L**2 + 1)

    def state_to_lattice(self):
        """
        Convert state to lattice [0,1] -> [-1,1]
        """
        lattice = np.reshape(2 * self.state - 1, (self.L, self.L))
        return lattice

    def compute_energy(self):
        # J=0 except for nearest neighbor
        lattice = self.state_to_lattice()
        energy = -self.J * sum(
            lattice[i, j] * (lattice[i - 1, j] + lattice[i, j - 1])
            for i in range(self.L)
            for j in range(self.L)
        )
        return energy

    def step(self, action):
        info = {}

        if action == self.L**2:
            # finish action
            reward = -self.compute_energy()
            done = True
        else:
            self.state[action] = 1 if self.state[action] == 0 else 0
            reward = 0
            done = False

        return self.state, float(reward), bool(done), info

    def reset(self):
        self.state = self.observation_space.sample()
        return self.state

    def render(self, mode="human"):
        print(f"{self.state_to_lattice()}")

    # def close(self):
    #   ...
