import numpy as np
import pytest

from pystochastic.utils import (
    ensure_single_arg_constant_function,
    generate_times,
    single_arg_constant_function,
)


@pytest.mark.parametrize(
    "end, n",
    [
        (
            1.0,
            16,
        )
    ],
)
def test_generate_times(end: float, n: int) -> None:
    ls = generate_times(end=end, n=n)

    assert (ls == np.linspace(0, end, n + 1)).all()


def test_single_arg_constant_function() -> None:
    const = single_arg_constant_function(value=4)

    assert callable(const)
    assert const(1) == 4


def test_ensure_single_arg_constant_function() -> None:
    const = ensure_single_arg_constant_function(value=4)

    assert callable(const)
    assert const(1) == 4

    func = ensure_single_arg_constant_function(value=lambda x: 5)

    assert callable(func)
    assert func(1) == 5
