from importlib.metadata import entry_points
from gym.envs.registration import register

register(
    id="xymodel-v0",
    entry_point="gym_xymodel.envs:XYmodelEnv",
)
register(
    id="isingmodel-v0",
    entry_point="gym_xymodel.envs:IsingModelEnv",
)

register(
    id="isingmodel1d-v0",
    entry_point="gym_xymodel.envs:IsingModel1DEnv",
)

register(
    id="isingmodel1dendrew-v0",
    entry_point="gym_xymodel.envs:IsingModel1DEndRewEnv",
)
