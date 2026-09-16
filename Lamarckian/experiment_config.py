"""Shared experiment budgets for the Evolution Gym benchmark tasks.

The values are copied from ``MACD/utils/MyUtils.py:get_par``. ``tc`` is the
maximum PPO update count for one robot and ``max_eva`` is the total number of
unique robots evaluated during a complete evolutionary run.
"""

from typing import Tuple


ENV_LIST = (
    "Walker-v0", "AreaMaximizer-v0", "Carrier-v0", "Thrower-v0",
    "UpStepper-v0", "ObstacleTraverser-v0", "ObstacleTraverser-v1",
    "GapJumper-v0", "BeamSlider-v0",
)

_BUDGETS = {
    "Walker-v0": (100, 500),
    "AreaMaximizer-v0": (100, 600),
    "Carrier-v0": (100, 500),
    "Thrower-v0": (150, 300),
    "UpStepper-v0": (150, 600),
    "ObstacleTraverser-v0": (150, 1000),
    "ObstacleTraverser-v1": (200, 1000),
    "GapJumper-v0": (200, 1000),
    "BeamSlider-v0": (200, 1000),
}


def get_par(env_name: str) -> Tuple[int, int]:
    """Return ``(max_eva, tc)`` exactly as defined by the MACD reference."""
    try:
        return _BUDGETS[env_name]
    except KeyError as error:
        raise ValueError(f"No MACD get_par budget is defined for {env_name!r}.") from error
