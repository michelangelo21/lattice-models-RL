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
        r: float = 1.0,
        D: npt.NDArray = np.array([0, 0, 1], dtype=np.float32),  # x y z
        step_size: float = 0.1,
        max_episode_steps: int = 16,
    ):
        # lattice side_len x side_len
        self.L = L
        self.J = J
        self.r = r
        self.D = D
        self.step_size = step_size
        self.max_episode_steps = max_episode_steps
        self.step_no = 0

        # self.observation_space = spaces.Box(
        #     low=0, high=255, shape=(1, side_len, side_len), dtype=np.uint8
        # ) # cnn
        highs = np.pi * np.ones((2, L, L))
        highs[0, :, :] *= 2
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
        # ! both angles are in [0, 2pi], so lattice is not unequivocal
        # lattice = np.reshape(2 * np.pi * self.state, (2, self.L, self.L))
        lattice = self.state
        return lattice

    def compute_energy(self):
        """
        Computes energy of the current state
        """
        # J=0 except for nearest neighbor
        angles = self.state_to_lattice()

        x = self.r * np.sin(angles[0, :, :]) * np.cos(angles[1, :, :])
        y = self.r * np.sin(angles[0, :, :]) * np.sin(angles[1, :, :])
        z = self.r * np.cos(angles[0, :, :])
        lattice_cartesian = np.stack((x, y, z), axis=2)

        # ! cross product does not commute, four directions cancel each other out
        energy = (
            -self.J
            * np.sum(
                np.dot(
                    np.cross(lattice_cartesian, np.roll(lattice_cartesian, -1, axis=0)),
                    self.D,
                )
                + np.dot(
                    np.cross(lattice_cartesian, np.roll(lattice_cartesian, -1, axis=1)),
                    self.D,
                )
            )
            / self.L**2
        )

        if energy < self.min_energy:
            self.min_energy = energy
            self.min_state = self.state
        return energy

    def step(self, action):
        # self.state = (self.state + self.step_size * action) % (2 * np.pi)
        self.state = self.state + self.step_size * action
        idx = self.state[1] > np.pi
        self.state[1][idx] = np.pi - self.state[1][idx]
        self.state[0][idx] += np.pi
        self.state[0] %= 2 * np.pi
        assert np.all(self.state[1] <= np.pi)

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
