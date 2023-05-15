import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


# TODO change name from Ladder to something else
class Ladder2DEnv(gym.Env):
    """
    Ladder2DEnv contains side_len x side_len lattice
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        L: int = 4,
        mu: float = 0.0,
        J: float = 1.0,
        step_size: float = 0.1 * 2 * np.pi,  # out of 2pi
        max_episode_steps: int = 16,
    ):
        # lattice side_len x side_len
        self.L = L
        self.mu = mu
        self.J = J
        self.step_size = step_size
        self.max_episode_steps = max_episode_steps
        self.step_no = 0

        # self.observation_space = spaces.Box(
        #     low=0, high=255, shape=(1, side_len, side_len), dtype=np.uint8
        # ) # cnn

        self.observation_space = spaces.Box(
            low=0, high=2 * np.pi, shape=(2, L, L), dtype=np.float32
        )

        self.state = self.observation_space.sample()

        t = 1
        H_kinetic = np.zeros((2 * L**2, 2 * L**2), dtype=np.float32)
        numbering = np.arange(L**2).reshape(L, L)
        for x in range(L):
            for y in range(L):
                i = numbering[x, y]  # i = x * L + y,
                j = numbering[x, y - 1]  # left
                H_kinetic[i, j] = t
                H_kinetic[j, i] = t
                j = numbering[x - 1, y]  # up
                H_kinetic[i, j] = t
                H_kinetic[j, i] = t

        H_chem = np.diag(mu * np.ones(2 * L**2, dtype=np.float32))
        self.H_base = -H_kinetic - H_chem

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
        Convert state (angles) to cartesian coordinates
        """
        # lattice = np.reshape(2 * np.pi * self.state, (2, self.L, self.L))
        angles = self.state
        x = np.sin(angles[0]) * np.cos(angles[1])
        y = np.sin(angles[0]) * np.sin(angles[1])
        z = np.cos(angles[0])
        lattice_cartesian = np.stack((x, y, z), axis=0).reshape(3, -1)
        return lattice_cartesian

    def compute_energy(self):
        """
        Computes energy of the current state
        """
        # J=0 except for nearest neighbor
        xyz = self.state_to_lattice()  # S_ij
        H_mag = np.zeros((2 * self.L**2, 2 * self.L**2), dtype=np.complex64)
        H_mag[: self.L**2, : self.L**2] = np.diag(xyz[2])
        H_mag[: self.L**2, self.L**2 :] = np.diag(xyz[0] - 1j * xyz[1])
        H_mag[self.L**2 :, : self.L**2] = np.diag(xyz[0] + 1j * xyz[1])
        H_mag[self.L**2 :, self.L**2 :] = np.diag(-xyz[2])

        Hamiltonian = self.H_base - self.J * H_mag
        evals = np.linalg.eigvalsh(Hamiltonian)
        energy = evals[evals < self.mu].sum() / self.L**2

        if energy < self.min_energy:
            self.min_energy = energy
            self.min_state = self.state
        return energy

    def step(self, action):
        self.state = (self.state + self.step_size * action) % (2 * np.pi)

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
