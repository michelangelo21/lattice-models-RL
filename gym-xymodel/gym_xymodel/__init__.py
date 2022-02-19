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
