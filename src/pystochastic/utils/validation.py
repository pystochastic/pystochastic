from inspect import signature
from typing import Any, Callable

import numpy as np
import numpy.typing as npt


def check_positive_integer(
    *,
    n: int,
    name: str = "",
) -> None:
    """Ensure that the number is a positive integer."""
    if not isinstance(n, int):
        raise TypeError(f"{name} must be an integer; got {type(n).__name__}.")
    if n <= 0:
        raise ValueError(f"{name} must be positive; got {n}.")


def check_nonnegative_integer(
    *,
    n: int,
    name: str = "",
) -> None:
    """Ensure that the number is a nonnegative integer."""
    if not isinstance(n, int):
        raise TypeError(f"{name} must be an integer; got {type(n).__name__}.")
    if n < 0:
        raise ValueError(f"{name} must be nonnegative; got {n}.")


def check_numeric(
    *,
    value: float | int,
    name: str = "",
) -> None:
    """Ensure that the value is numeric."""
    if not isinstance(value, (float, int)):
        raise TypeError(f"{name} value must be a number; got {type(value).__name__}.")


def check_positive_number(
    *,
    value: float,
    name: str = "",
) -> None:
    """Ensure that the value is a positive number."""
    check_numeric(value=value, name=name)
    if value <= 0:
        raise ValueError(f"{name} value must be positive; got {value}.")


def check_nonnegative_number(
    *,
    value: float,
    name: str = "",
) -> None:
    """Ensure that the value is a nonnegative number"""
    check_numeric(value=value, name=name)
    if value < 0:
        raise ValueError(f"{name} value must be nonnegative; got {value}.")


def check_increments(
    *,
    times: npt.NDArray[np.float64] | list[float],
) -> npt.NDArray[np.float64]:
    """Ensure a positive, monotonically increasing sequence."""
    increments = np.diff(times)

    if np.any([t < 0 for t in times]):
        raise ValueError("Times must be nonnegative.")

    if np.any([t <= 0 for t in increments]):
        raise ValueError("Times must be strictly increasing.")

    return increments


def times_to_increments(
    *,
    times: npt.NDArray[np.float64] | list[float],
) -> npt.NDArray[np.float64]:
    """Ensure a positive, monotonically increasing sequence."""
    return check_increments(times=times)


def check_numeric_or_single_arg_callable(
    *,
    value: Callable[[Any], float | int] | float | int,
    name: str = "",
) -> None:
    """Ensure a numeric of single arg callable."""
    if callable(value) and len(signature(value).parameters) != 1:
        raise ValueError(
            f"{name} callable must have a single argument; got {len(signature(value).parameters)}."
        )
    if not isinstance(value, (float, int)) and not callable(value):
        raise TypeError(
            f"{name} must be numeric or a single argument callable; got {type(value).__name__}.",
        )
