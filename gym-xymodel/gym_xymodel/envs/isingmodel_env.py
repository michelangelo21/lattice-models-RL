import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

SIDE_LENGTH = 5

class IsingModelEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self):
    # lattice SIDE_LENGTH x SIDE_LENGTH
    self.observation_space = spaces.MultiBinary([SIDE_LENGTH, SIDE_LENGTH])
    # states are -1 or 1
    self.lattice = self.observation_space.sample() * 2 - 1
    self.energy = self.compute_energy()
    self.action_space = spaces.MultiDiscrete([SIDE_LENGTH, SIDE_LENGTH])

  def compute_energy(self):
    # J=0 except for nearest neighbor
    energy = - sum(self.lattice[i,j] * (self.lattice[i-1,j] + self.lattice[i,j-1]) for i in range(SIDE_LENGTH) for j in range(SIDE_LENGTH))
    return energy

  def finish_condition(self):
    if np.all(self.lattice == self.lattice[0,0]):
      return True
    else:
      return False

  def step(self, action):
    self.lattice[action[0], action[1]] *= -1
    new_energy = self.compute_energy()
    # finding minimal energy state, reward is difference between old and new energy
    reward = - (new_energy - self.energy)
    self.energy = new_energy

    done = self.finish_condition()
    info = {}
    return (self.lattice+1)//2, reward, done, info # state, reward, done, info

  def reset(self):
    self.lattice = self.observation_space.sample() * 2 - 1
    return (self.lattice+1)//2

  # def render(self, mode='human'):
  #   ...
  # def close(self):
  #   ...