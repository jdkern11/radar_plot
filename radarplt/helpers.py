from typing import List


def minmax_scale(val: float, scale: List[float]) -> float:
    """Scales value to be between scale[0] and scale[1]

    Args:
        val: Value to scale
        scale: List with index 0 corresponding to min value and 1 corresponding to max
    """
    return (val - scale[0]) / (scale[1] - scale[0])
