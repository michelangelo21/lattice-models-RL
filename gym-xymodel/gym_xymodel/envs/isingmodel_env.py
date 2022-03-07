import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class IsingModelEnv(gym.Env):
    """
    IsingModelEnv contains SIDE_LENGTH x SIDE_LENGTH lattice
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, side_len: int = 4):
        """Initialization of the gym environment"""
        # lattice side_len x side_len
        self.side_len = side_len

        self.observation_space = spaces.MultiBinary(self.side_len**2)
        # states are -1 or 1
        self.state = self.observation_space.sample()
        self.energy = self.compute_energy()
        self.action_space = spaces.Discrete(self.side_len**2)

    def state_to_lattice(self):
        """
        Convert state to lattice [0,1] -> [-1,1]
        """
        lattice = np.reshape(2 * self.state - 1, (self.side_len, self.side_len))
        return lattice

    def compute_energy(self):
        # J=0 except for nearest neighbor
        lattice = self.state_to_lattice()
        energy = -sum(
            lattice[i, j] * (lattice[i - 1, j] + lattice[i, j - 1])
            for i in range(self.side_len)
            for j in range(self.side_len)
        )
        return energy

    def finish_condition(self):
        """
        Finish condition is when energy is minimal, and all spins are the same
        """
        if np.all(self.state == self.state[0]):
            return True
        else:
            return False

    def step(self, action):
        self.state[action] = 1 if self.state[action] == 0 else 0
        new_energy = self.compute_energy()
        # finding minimal energy state, reward is difference between old and new energy
        reward = -(new_energy - self.energy)
        self.energy = new_energy

        # Done when energy is minimal, and all spins are the same
        done = np.all(self.state == self.state[0])
        info = {"energy": self.energy}
        return self.state, reward, done, info

    def reset(self):
        self.state = self.observation_space.sample()
        return self.state

    def render(self, mode="human"):
        print(f"{self.state_to_lattice()}")

    # def close(self):
    #   ...
