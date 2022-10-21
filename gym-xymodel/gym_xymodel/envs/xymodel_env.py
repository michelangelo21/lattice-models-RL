import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class XYmodelEnv(gym.Env):
    """
    XYModelEnv contains side_len x side_len lattice
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, side_len: int = 4):
        # lattice side_len x side_len
        self.side_len = side_len

        # self.observation_space = spaces.Box(
        #     low=0, high=255, shape=(1, side_len, side_len), dtype=np.uint8
        # ) # cnn

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(side_len**2,), dtype=np.uint8
        )  # mlp

        self.state = self.observation_space.sample()
        self.energy = self.compute_energy()

        # self.action_space = spaces.Box(
        #     low=-5,
        #     high=5,
        #     shape=(1, self.side_len, self.side_len),
        #     dtype=np.int8,
        # ) # cnn not working
        self.action_space = spaces.Box(
            low=-5,
            high=5,
            shape=(self.side_len**2,),
            dtype=np.int8,
        )  # mlp

        # self.action_space = spaces.Box(
        #     low=-1, high=1, shape=(self.side_len**2,), dtype=np.float32
        # )

    def state_to_lattice(self):
        """
        Convert state to lattice [-1,1] -> [-pi/2,pi/2]
        """
        lattice = np.reshape(
            2 * np.pi / 256 * self.state, (self.side_len, self.side_len)
        )
        return lattice

    def compute_energy(self):
        """
        Computes energy of the current state
        """
        # J=0 except for nearest neighbor
        lattice = self.state_to_lattice()
        energy = -sum(
            np.cos(lattice[i, j] - lattice[i - 1, j])
            + np.cos(lattice[i, i] - lattice[i, j - 1])
            for i in range(self.side_len)
            for j in range(self.side_len)
        )
        return energy

    def step(self, action):
        # self.state = (self.state + 1 + action) % 2 - 1
        # self.state[choosen_spin] = (
        #     self.state[choosen_spin] + 1 + choosen_rotation
        # ) % 2 - 1
        self.state += np.uint8(action)
        new_energy = self.compute_energy()
        # finding minimal energy state, reward is difference between old and new energy
        reward = -(new_energy - self.energy)
        self.energy = new_energy

        done = bool(np.all(np.abs(self.state - self.state[0]) < 1e-6))
        info = {"energy": self.energy}
        return self.state, reward, done, info  # state, reward, done, info

    def reset(self):
        self.state = self.observation_space.sample()
        return self.state

    def render(self, mode="human"):
        print(f"{self.state_to_lattice()}")

    # def close(self):
    #     ...
