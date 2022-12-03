import numpy as np
from typing import Callable


def cosine_schedule(
    eta_min: float, eta_max: float, n_cycles: int
) -> Callable[[float], float]:
    """
    Cosine learning rate schedule.
    """

    def cos_decay(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining == 0:
            return eta_min
        else:
            return eta_min + (eta_max - eta_min) * 0.5 * (
                1 + np.cos(np.pi * ((n_cycles * (1 - progress_remaining)) % 1))
            )

    return cos_decay
