import gym


class ContinuousLearningWrapper(gym.Wrapper):
    """
    Returns energy as reward every step
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = -self.env.compute_energy()
        return obs, reward, done, info
