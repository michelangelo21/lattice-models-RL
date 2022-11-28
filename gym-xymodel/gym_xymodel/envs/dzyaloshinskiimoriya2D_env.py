import numpy as np
import numpy.typing as npt
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class DzyaloshinskiiMoriya2DEnv(gym.Env):
    """
    XYModelEnv contains side_len x side_len lattice
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        L: int = 4,
        J: float = 1.0,
        D: float = 1.0,
        step_size: float = 1.0,  # out of pi, 2pi
        max_episode_steps: int = 16,
    ):
        # lattice side_len x side_len
        self.L = L
        self.J = J
        self.D = D
        assert step_size < np.pi
        self.step_size = step_size
        self.max_episode_steps = max_episode_steps
        self.step_no = 0

        # self.observation_space = spaces.Box(
        #     low=0, high=255, shape=(1, side_len, side_len), dtype=np.uint8
        # ) # cnn
        highs = np.pi * np.ones((2, L, L))
        highs[1, :, :] *= 2
        self.observation_space = spaces.Box(
            low=np.zeros((2, L, L)),
            high=highs,
            shape=(2, L, L),
            dtype=np.float32,
        )

        self.state = self.observation_space.sample()

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2, L, L),
            dtype=np.float32,
        )

        self.min_energy = float("inf")
        self.min_state = None

    def state_to_lattice(self):
        """
        Convert state to lattice [0, 1] -> [0, 2pi]
        """
        # lattice = np.reshape(2 * np.pi * self.state, (2, self.L, self.L))
        angles = self.state
        x = np.sin(angles[0, :, :]) * np.cos(angles[1, :, :])
        y = np.sin(angles[0, :, :]) * np.sin(angles[1, :, :])
        z = np.cos(angles[0, :, :])
        lattice_cartesian = np.stack((x, y, z), axis=2)
        return lattice_cartesian

    def compute_energy(self):
        """
        Computes energy of the current state
        """
        # J=0 except for nearest neighbor
        lattice_cartesian = self.state_to_lattice()

        # ! cross product does not commute, four directions cancel each other out
        lat_roll_i = np.roll(lattice_cartesian, 1, axis=0)
        lat_roll_j = np.roll(lattice_cartesian, 1, axis=1)
        energy = (
            -self.J
            * (
                (lattice_cartesian * lat_roll_i).sum()
                + (lattice_cartesian * lat_roll_j).sum()
            )
            - self.D
            * (
                np.cross(lattice_cartesian, lat_roll_i)[:, :, 1].sum()  # y
                + np.cross(lattice_cartesian, lat_roll_j)[:, :, 0].sum()  # x
            )
        ) / self.L**2

        if energy < self.min_energy:
            self.min_energy = energy
            self.min_state = self.state
        return energy

    def step(self, action):
        # self.state = (self.state + self.step_size * action) % (2 * np.pi)
        self.state = self.state + self.step_size * action
        idx_greater = self.state[0] > np.pi
        idx_less = self.state[0] < 0
        self.state[0][idx_greater] = 2 * np.pi - self.state[0][idx_greater]
        self.state[0][idx_less] *= -1
        self.state[1][idx_greater | idx_less] += np.pi
        self.state[1] %= 2 * np.pi

        assert np.all(self.state[0] <= np.pi)
        assert np.all(0 <= self.state)

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
