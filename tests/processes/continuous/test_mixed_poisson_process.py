from typing import Any, Callable

import numpy as np
import pytest

from pystochastic.processes.continuous import MixedPoissonProcess as TestType


@pytest.mark.parametrize(
    "rate_func, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
        (
            -1,
            TypeError,
        ),
    ],
)
def test_mixed_poisson_process_invalid_rate_func(
    rate_func: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(rate_func=rate_func)


@pytest.mark.parametrize(
    "rate_args, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
        (
            -1,
            TypeError,
        ),
    ],
)
def test_mixed_poisson_process_invalid_rate_args(
    rate_args: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(rate_func=lambda x: x, rate_args=rate_args)


@pytest.mark.parametrize(
    "rate_kwargs, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
        (
            -1,
            TypeError,
        ),
    ],
)
def test_mixed_poisson_process_invalid_rate_kwargs(
    rate_kwargs: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(rate_func=lambda x: x, rate_kwargs=rate_kwargs)


def test_mixed_poisson_process_str_repr() -> None:
    instance = TestType(rate_func=lambda x: x, rate_args=(1.0,), rate_kwargs={})

    assert str(instance) == "Mixed Poisson process with random rate."
    assert isinstance(repr(instance), str)


@pytest.mark.parametrize(
    "func, func_args, n, expected",
    [
        (
            lambda x: x,
            (1.0,),
            5,
            [0.0, 1.887763, 2.25833, 3.567758, 4.02236, 5.545158],
        ),
        (
            lambda x: x * x,
            (5.0,),
            5,
            [
                0.0,
                0.07551051031630401,
                0.09033318432261209,
                0.14271030458463418,
                0.16089441934628196,
                0.22180631402493253,
            ],
        ),
    ],
)
def test_poisson_process_sample_values(
    func: Callable[..., float],
    func_args: tuple[float, ...],
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        rate_func=func,
        rate_args=func_args,
        rng=rng,
    )

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "func, func_args, length, expected",
    [
        (
            lambda x: x,
            (0.5,),
            5.0,
            [0.0, 3.775526, 4.516659, 7.135515],
        ),
        (
            lambda x: x * x,
            (0.8,),
            5.0,
            [0.0, 2.949629, 3.52864, 5.574621],
        ),
    ],
)
def test_poisson_process_sample_with_length(
    func: Callable[..., float],
    func_args: tuple[float, ...],
    length: float,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        rate_func=func,
        rate_args=func_args,
        rng=rng,
    )

    samples = instance.sample_with_length(length=length)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)
