from importlib.metadata import entry_points
from gym.envs.registration import register

register(
    id="xymodel-v0",
    entry_point="gym_xymodel.envs:XYmodelEnv",
)
register(
    id="ising2d-v0",
    entry_point="gym_xymodel.envs:Ising2DEnv",
)

register(
    id="isingmodel1d-v0",
    entry_point="gym_xymodel.envs:IsingModel1DEnv",
)

register(
    id="isingmodel1dendrew-v0",
    entry_point="gym_xymodel.envs:IsingModel1DEndRewEnv",
)

register(
    id="falicovkimball1D-v0",
    entry_point="gym_xymodel.envs:FalicovKimball1DEnv",
)

register(
    id="falicovkimball2D-v0",
    entry_point="gym_xymodel.envs:FalicovKimball2DEnv",
)
