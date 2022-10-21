import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class FalicovKimball1DEnv(gym.Env):
    """
    FalicovKimball1DEnv contains SIDE_LENGTH lattice
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        L: int = 8,
        Ne: int = 4,
        t: float = 1.0,
        U: float = 2.0,
        max_steps: int = 4,
        is_PBC: bool = True,
        seed=None,
    ):
        """Initialization of the gym environment"""
        self.L = L  # lattice side_length
        self.Ne = Ne
        self.t = t
        self.U = U
        self.max_steps = max_steps
        self.step_no = 1
        self.is_PBC = is_PBC
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.MultiBinary(self.L)
        # states are 0 or 1
        self.state = self.random_state()
        self.energy = self.compute_energy()
        self.action_space = spaces.Discrete(
            self.L + 1
        )  # +1 for pass action (end episode)

    def random_state(self):
        pos = self.rng.choice(self.L, size=self.Ne, replace=False)
        lattice = np.full(self.L, False)
        lattice[pos] = True
        return lattice

    def compute_energy(self):
        """Computes energy of the current state per node."""
        above_diag = np.diag(self.t * np.ones(self.L - 1), k=1)
        below_diag = np.diag(np.conj(self.t) * np.ones(self.L - 1), k=-1)
        H_kinetic = above_diag + below_diag
        if self.is_PBC:
            H_kinetic[0, -1] = np.conj(self.t)
            H_kinetic[-1, 0] = self.t

        H = -H_kinetic - np.diag(self.U * self.state)

        w, _ = np.linalg.eigh(H)

        energy = np.sum(w[: self.Ne])
        return energy / self.L

    def step(self, action):
        info = {}

        # if action == self.L, skip round and do nothing,
        # this prevents agent from bouncing from the ground state
        # ? maybe done = True, and end round
        if action == self.L:
            reward = -self.compute_energy()
            done = True
            return self.state, float(reward), bool(done), info
        else:
            self.state[action] = 1 if self.state[action] == 0 else 0

        self.step_no += 1
        if self.step_no > self.max_steps:
            done = True
            reward = -self.compute_energy()
        else:
            done = False
            reward = 0

        return self.state, float(reward), bool(done), info

    def reset(self):
        self.state = self.observation_space.sample()
        self.step_no = 1
        return self.state

    def render(self, mode="human"):
        print(f"{self.state_to_lattice()}")

    # def close(self):
    #   ...
