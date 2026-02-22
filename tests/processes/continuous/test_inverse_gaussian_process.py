from typing import Any, Callable

import numpy as np
import pytest

from pystochastic.processes.continuous import InverseGaussianProcess as TestType


@pytest.mark.parametrize(
    "mean_func, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
        (
            1,
            TypeError,
        ),
        (
            lambda x, y: x + y,
            ValueError,
        ),
        (
            lambda: 42,
            ValueError,
        ),
    ],
)
def test_inverse_gaussian_process_invalid_mean_func(
    mean_func: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(mean=mean_func)


@pytest.mark.parametrize(
    "scale, error_type",
    [
        (
            -1,
            ValueError,
        ),
        (
            0,
            ValueError,
        ),
        (
            "invalid",
            TypeError,
        ),
    ],
)
def test_inverse_gaussian_process_invalid_scale(scale: Any, error_type: type) -> None:
    with pytest.raises(error_type):
        TestType(scale=scale)


def test_inverse_gaussian_process_str_repr() -> None:
    instance = TestType(mean=lambda x: 2 * x, scale=1.5, t=1.0)

    assert (
        str(instance) == "Inverse Gaussian process with mean <lambda> "
        "and scale 1.5 on interval [0, 1.0]."
    )

    assert repr(instance) == "InverseGaussianProcess(mean=<lambda>, scale=1.5, t=1.0)"


def test_inverse_gaussian_process_sample_shape() -> None:
    instance = TestType(mean=lambda x: 2 * x, scale=1.5, t=1.0)
    n = 1000
    samples = instance.sample(n)

    assert samples.shape == (n + 1,)

    t = np.linspace(0, 1.0, n + 1)
    samples_at = instance.sample_at(times=t)

    assert samples_at.shape == (n + 1,)


@pytest.mark.parametrize(
    "mean_func, scale, t, n, expected",
    [
        (
            lambda x: 2 * x,
            1.5,
            2.0,
            5,
            [
                0.0,
                1.716619,
                1.699605,
                1.66945,
                1.764193,
                1.462059,
            ],
        ),
        (
            lambda x: 3 * x,
            2.0,
            1.0,
            5,
            [
                0.0,
                1.375527,
                1.358391,
                1.524974,
                1.769196,
                1.7714,
            ],
        ),
    ],
)
def test_inverse_gaussian_process_sample_values(
    mean_func: Callable[[float], float],
    scale: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        mean=mean_func,
        scale=scale,
        t=t,
        rng=rng,
    )

    samples = instance.sample(n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "mean_func, scale, t, n, expected",
    [
        (
            lambda x: 2 * x,
            1.5,
            2.0,
            5,
            [
                0.0,
                1.716619,
                1.699605,
                1.66945,
                1.764193,
                1.462059,
            ],
        ),
        (
            lambda x: 3 * x,
            2.0,
            1.0,
            5,
            [
                0.0,
                1.375527,
                1.358391,
                1.524974,
                1.769196,
                1.7714,
            ],
        ),
    ],
)
def test_inverse_gaussian_process_sample_at_values(
    mean_func: Callable[[float], float],
    scale: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        mean=mean_func,
        scale=scale,
        t=t,
        rng=rng,
    )

    times = np.linspace(0, t, n + 1)
    samples_at = instance.sample_at(times)
    print(samples_at)
    np.testing.assert_allclose(samples_at, expected, rtol=1e-6)
