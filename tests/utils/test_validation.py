from typing import Any, Callable

import pytest

from pystochastic.utils.validation import (
    check_increments,
    check_nonnegative_integer,
    check_nonnegative_number,
    check_numeric,
    check_numeric_or_single_arg_callable,
    check_positive_integer,
    check_positive_number,
)


@pytest.mark.parametrize(
    "n",
    [
        4.2,
        "invalid",
    ],
)
def test_check_positive_integer_failure_type(n: Any) -> None:
    with pytest.raises(TypeError):
        check_positive_integer(n=n)


@pytest.mark.parametrize(
    "n",
    [
        -1,
        0,
    ],
)
def test_check_positive_integer_failure_value(n: int) -> None:
    with pytest.raises(ValueError):
        check_positive_integer(n=n)


@pytest.mark.parametrize(
    "n",
    [
        4,
        1,
    ],
)
def test_check_positive_integer(n: int) -> None:
    check_positive_integer(n=n)


@pytest.mark.parametrize(
    "n",
    [
        4.2,
        "invalid",
    ],
)
def test_check_nonnegative_integer_failure_type(n: Any) -> None:
    with pytest.raises(TypeError):
        check_nonnegative_integer(n=n)


@pytest.mark.parametrize(
    "n",
    [
        -1,
    ],
)
def test_check_nonnegative_integer_failure_value(n: int) -> None:
    with pytest.raises(ValueError):
        check_nonnegative_integer(n=n)


@pytest.mark.parametrize(
    "n",
    [
        4,
        0,
    ],
)
def test_check_nonnegative_integer(n: int) -> None:
    check_nonnegative_integer(n=n)


@pytest.mark.parametrize(
    "value",
    [
        "invalid",
    ],
)
def test_check_numeric_failure_type(value: Any) -> None:
    with pytest.raises(TypeError):
        check_numeric(value=value)


@pytest.mark.parametrize(
    "value",
    [
        4,
        0,
        -4.2,
    ],
)
def test_check_numeric(value: float) -> None:
    check_numeric(value=value)


@pytest.mark.parametrize(
    "value",
    [
        "invalid",
    ],
)
def test_check_positive_number_failure_type(value: Any) -> None:
    with pytest.raises(TypeError):
        check_positive_number(value=value)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1.2,
    ],
)
def test_check_positive_number_failure_value(value: float) -> None:
    with pytest.raises(ValueError):
        check_positive_number(value=value)


@pytest.mark.parametrize(
    "value",
    [
        4,
        1,
    ],
)
def test_check_positive_number(value: float) -> None:
    check_positive_number(value=value)


@pytest.mark.parametrize(
    "value",
    [
        "invalid",
    ],
)
def test_check_nonnegative_number_failure_type(value: Any) -> None:
    with pytest.raises(TypeError):
        check_nonnegative_number(value=value)


@pytest.mark.parametrize(
    "value",
    [
        -2,
        -1.2,
    ],
)
def test_check_nonnegative_number_failure_value(value: float) -> None:
    with pytest.raises(ValueError):
        check_nonnegative_number(value=value)


@pytest.mark.parametrize(
    "value",
    [
        4,
        1,
        0,
    ],
)
def test_check_nonnegative_number(value: float) -> None:
    check_nonnegative_number(value=value)


@pytest.mark.parametrize(
    "value",
    [
        lambda x, y: 5,
        "test",
    ],
)
def test_check_numeric_or_single_arg_callable_invalid(value: Any) -> None:
    with pytest.raises(ValueError):
        check_numeric_or_single_arg_callable(value=value)


@pytest.mark.parametrize(
    "value",
    [
        5,
        lambda x: 5,
    ],
)
def test_check_numeric_or_single_arg_callable(
    value: Callable[[Any], float | int] | float | int,
) -> None:

    check_numeric_or_single_arg_callable(value=value)


@pytest.mark.parametrize(
    "times",
    [
        [-1, 0, 1],
        [0, 2, 1],
    ],
)
def test_check_increments_invalid(times: list[float]) -> None:
    with pytest.raises(ValueError):
        check_increments(times=times)


@pytest.mark.parametrize(
    "times",
    [
        [0, 1, 2],
    ],
)
def test_check_increments(times: list[float]) -> None:
    check_increments(times=times)
