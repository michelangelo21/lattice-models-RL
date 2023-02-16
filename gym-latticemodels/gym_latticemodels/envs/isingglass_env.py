import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class IsingGlassEnv(gym.Env):
    """
    Ising2DEnv contains SIDE_LENGTH x SIDE_LENGTH lattice
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, L: int = 4):
        """Initialization of the gym environment"""
        # lattice side_len x side_len
        self.L = L

        # self.J = np.ones((2, L, L), dtype=np.float32)
        # self.J = 2 * np.random.rand(2, L, L).astype(np.float32) - 1
        self.J = np.array(
            [
                [
                    [-0.12437874, -0.7693427, -0.09819233, 0.70126045],
                    [0.2832359, 0.40257168, 0.606985, -0.72242725],
                    [-0.93177474, 0.38692272, 0.23516941, 0.9015126],
                    [-0.43586504, -0.42629552, -0.32832438, -0.07254606],
                ],
                [
                    [0.97304547, -0.55682, -0.4540189, -0.39269322],
                    [0.7138872, 0.05342662, -0.44557214, -0.50881565],
                    [0.68253195, 0.79052234, 0.18135059, -0.3524053],
                    [-0.84415865, -0.7448032, -0.29852146, 0.9696255],
                ],
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(3, self.L, self.L)
        )  # spins, J_left, J_up
        # spins are -1 or 1
        self.state = np.concatenate(
            (np.ones((1, L, L)), self.J), dtype=np.float32
        )  # spins will be reset in reset()
        self.action_space = spaces.Discrete(self.L**2 + 1)

        self.min_energy = float("inf")
        self.min_state = None

    # def state_to_lattice(self):
    #     """
    #     Convert state to lattice [0,1] -> [-1,1]
    #     """
    #     lattice = np.reshape(2 * self.state - 1, (self.L, self.L))
    #     return lattice

    def compute_energy(self):
        # J=0 except for nearest neighbor
        # lattice = self.state_to_lattice()
        energy = (
            np.sum(self.state[1] * self.state[0] * np.roll(self.state[0], 1, axis=-1))
            + np.sum(self.state[2] * self.state[0] * np.roll(self.state[0], 1, axis=-2))
        ) / self.L**2

        if energy < self.min_energy:
            self.min_energy = energy
            self.min_state = self.state
        return energy

    def step(self, action):
        info = {}

        if action == self.L**2:
            # finish action
            reward = -self.compute_energy()
            done = True
        else:
            row = action // self.L
            col = action % self.L
            self.state[0][row, col] *= -1
            reward = 0
            done = False

        return self.state, float(reward), bool(done), info

    def reset(self):
        self.state[0] = 2 * np.random.randint(2, size=(1, self.L, self.L)) - 1
        return self.state

    def render(self, mode="human"):
        print(f"{self.state_to_lattice()}")

    # def close(self):
    #   ...
