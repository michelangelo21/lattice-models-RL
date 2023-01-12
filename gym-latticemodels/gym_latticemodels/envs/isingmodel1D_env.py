import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class IsingModel1DEnv(gym.Env):
    """
    IsingModel1DEnv contains SIDE_LENGTH lattice
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, L: int = 4):
        """Initialization of the gym environment"""
        # lattice side_len x side_len
        self.L = L

        self.observation_space = spaces.MultiBinary(self.L)
        # states are -1 or 1
        self.state = self.observation_space.sample()
        self.energy = self.compute_energy()
        self.action_space = spaces.Discrete(self.L)

    def state_to_lattice(self):
        """
        Convert state to lattice [0,1] -> [-1,1]
        """
        lattice = 2 * self.state - 1
        return lattice

    def compute_energy(self):
        """Computes energy of the current state per node (without optimalization yet"""
        # J=0 except for nearest neighbor
        lattice = self.state_to_lattice()
        energy = -sum(lattice[i] * (lattice[i - 1]) for i in range(self.L))
        return energy / self.L

    def finish_condition(self):
        """
        Finish condition is when energy is minimal, and all spins are the same.
        This is generally not known in other models.
        """
        return np.all(self.state == self.state[0])

    def step(self, action):
        self.state[action] = 1 if self.state[action] == 0 else 0
        new_energy = self.compute_energy()
        # finding minimal energy state, reward is difference between old and new energy
        reward = -(new_energy - self.energy)
        self.energy = new_energy

        # Done when energy is minimal, and all spins are the same
        done = np.all(self.state == self.state[0])
        info = {"energy": self.energy}
        return self.state, float(reward), bool(done), info

    def reset(self):
        self.state = self.observation_space.sample()
        return self.state

    def render(self, mode="human"):
        print(f"{self.state_to_lattice()}")

    # def close(self):
    #   ...
