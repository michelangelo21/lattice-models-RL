import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

SIDE_LENGTH = 3

class XYmodelEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self):
    # lattice SIDE_LENGTH x SIDE_LENGTH
    self.observation_space = spaces.Box(0, np.pi, (SIDE_LENGTH, SIDE_LENGTH), dtype=np.float32)
    self.state = self.observation_space.sample()
    self.energy = self.compute_energy(self.state)
    # self.lattice = self.rng.uniform(-np.pi/2, np.pi/2, (SIDE_LENGTH, SIDE_LENGTH))
    # self.action_space = spaces.Tuple(( # ((x,y), [changes...])
    #   spaces.Tuple((
    #     spaces.Discrete(SIDE_LENGTH),
    #     spaces.Discrete(SIDE_LENGTH)
    #   )),
    #   spaces.Box(-np.pi,np.pi,(SIDE_LENGTH,SIDE_LENGTH))
    # ))
    self.action_space = spaces.Box(-np.pi/2, np.pi/2, (SIDE_LENGTH*SIDE_LENGTH,), dtype=np.float32)

  def compute_energy(self, lattice):
    energy = 0.0
    for i in range(SIDE_LENGTH):
      for j in range(SIDE_LENGTH):
        # J=0 except for nearest neighbor
        energy -= np.cos(lattice[i,j] - lattice[i-1,j]) + np.cos(lattice[i,i] - lattice[i,j-1])
    return energy

  def step(self, action):
    self.state = (self.state + action.reshape(SIDE_LENGTH,SIDE_LENGTH)) % np.pi
    new_energy = self.compute_energy(self.state)
    # finding minimal energy state, reward is difference between old and new energy
    reward = - (new_energy - self.energy)
    self.energy = new_energy

    info = {}
    return self.state, reward, False, info # state, reward, done, info

  def reset(self):
    self.state = self.observation_space.sample()
    return self.state

  # def render(self, mode='human'):
  #   ...
  # def close(self):
  #   ...