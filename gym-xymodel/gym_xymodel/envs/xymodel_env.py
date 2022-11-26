import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class XY2DEnv(gym.Env):
    """
    XYModelEnv contains side_len x side_len lattice
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        L: int = 4,
        J: float = 1.0,
        step_size: float = 0.1,
        max_episode_steps: int = 16,
    ):
        # lattice side_len x side_len
        self.L = L
        self.J = J
        self.step_size = step_size
        self.max_episode_steps = max_episode_steps
        self.step_no = 0

        # self.observation_space = spaces.Box(
        #     low=0, high=255, shape=(1, side_len, side_len), dtype=np.uint8
        # ) # cnn

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(L**2,), dtype=np.float32
        )  # mlp

        self.state = self.observation_space.sample()

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(L**2,),
            dtype=np.float32,
        )

        self.min_energy = float("inf")
        self.min_state = None

    def state_to_lattice(self):
        """
        Convert state to lattice [0, 1] -> [0, pi]
        """
        lattice = np.reshape(2 * np.pi * self.state, (self.L, self.L))
        return lattice

    def compute_energy(self):
        """
        Computes energy of the current state
        """
        # J=0 except for nearest neighbor
        lattice = self.state_to_lattice()
        energy = (
            -self.J
            * np.sum(
                np.cos(lattice - np.roll(lattice, -1, axis=0))
                + np.cos(lattice - np.roll(lattice, -1, axis=1))
            )
            / self.L**2
        )

        if energy < self.min_energy:
            self.min_energy = energy
            self.min_state = self.state
        return energy

    def step(self, action):
        self.state = (self.state + self.step_size * action) % 1

        self.step_no += 1
        if self.step_no >= self.max_episode_steps:
            reward = -self.compute_energy()
            done = True
        else:
            reward = 0
            done = False

        info = {}
        return self.state, reward, done, info  # state, reward, done, info

    def reset(self):
        self.state = self.observation_space.sample()
        self.step_no = 0
        return self.state

    def render(self, mode="human"):
        print(f"{self.state_to_lattice()}")

    # def close(self):
    #     ...
