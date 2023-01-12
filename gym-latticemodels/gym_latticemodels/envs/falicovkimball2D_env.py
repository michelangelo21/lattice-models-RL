import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class FalicovKimball2DEnv(gym.Env):
    """
    FalicovKimball1DEnv contains SIDE_LENGTH lattice
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        L: int = 4,
        Ne: int = 8,
        t: float = 1.0,
        U: float = 2.0,
        max_steps: int = 16,
        # isPBC: bool = True,
    ):
        """Initialization of the gym environment"""
        self.L = L  # lattice side_length
        self.Ne = Ne
        self.t = t
        self.U = U
        self.max_steps = max_steps
        self.step_no = 1
        self.isPBC = True  # TODO add case without PBC

        # kinetic part of the Hamiltonian doesn't depend on the state
        H_kinetic = np.zeros((L**2, L**2), dtype=np.float32)
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

        self.H_kinetic = H_kinetic

        self.observation_space = spaces.MultiBinary(self.L**2)
        # states are 0 or 1
        self.state = self.random_state()
        # self.energy = self.compute_energy()
        self.action_space = spaces.MultiDiscrete(
            [self.L**2, self.L**2, 2]
        )  # third for pass action (end episode)

    def random_state(self):
        pos = np.random.choice(self.L**2, size=self.Ne, replace=False)
        lattice = np.full(self.L**2, False)
        lattice[pos] = True
        return lattice

    def compute_energy(self):
        """Computes energy of the current state per node."""
        H = -self.H_kinetic - np.diag(self.U * self.state)

        w, _ = np.linalg.eigh(H)

        energy = np.sum(w[: self.Ne])
        return energy / self.L**2

    def step(self, action):
        info = {}

        # finish round and do nothing,
        # this prevents agent from bouncing from the ground state
        if action[2] and self.step_no > self.max_steps // 2:
            reward = -self.compute_energy()
            done = True
            return self.state, float(reward), bool(done), info
        else:
            # ? maybe swap (symetries, different mask) - slower learning
            if self.state[action[0]] and not self.state[action[1]]:
                self.state[action[0]] = 0
                self.state[action[1]] = 1

            self.step_no += 1
            if self.step_no > self.max_steps:
                done = True
                reward = -self.compute_energy()
            else:
                done = False
                reward = 0

            return self.state, float(reward), bool(done), info

    def action_masks(self):
        # return np.hstack((self.state, np.logical_not(self.state)))
        if self.step_no <= self.max_steps // 2:
            return np.hstack((self.state, np.logical_not(self.state), [True, False]))
        else:
            return np.hstack((self.state, np.logical_not(self.state), [True, True]))

    def reset(self):
        self.state = self.random_state()
        self.step_no = 1
        return self.state

    def render(self, mode="human"):
        print(f"{self.state}")

    # def close(self):
    #   ...
