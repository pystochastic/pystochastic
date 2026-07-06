from typing import Any, Callable

import numpy as np
import numpy.typing as npt


def generate_times(
    *,
    end: float,
    n: int,
) -> npt.NDArray[np.float64]:
    """Generate a linspace from 0 to end for n increments."""
    return np.linspace(0, end, n + 1)


def single_arg_constant_function(
    *,
    value: float | int,
) -> Callable[[Any], float | int]:
    """Generate a single argument function which returns a constant value."""
    return lambda x: value


def ensure_single_arg_constant_function(
    *,
    value: Callable[[Any], float | int] | float | int,
) -> Callable[[Any], float | int]:
    """Convert the passed value into a const function if not one already."""
    if not callable(value):
        return single_arg_constant_function(value=value)
    return value
