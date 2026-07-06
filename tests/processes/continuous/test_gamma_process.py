from typing import Any

import numpy as np
import pytest

from pystochastic.processes.continuous import GammaProcess as TestType


@pytest.mark.parametrize(
    "mean, variance, rate, scale, error_type",
    [
        (None, None, None, None, ValueError),
        (1.0, None, None, None, ValueError),
        (None, 1.0, None, None, ValueError),
        (None, None, 1.0, None, ValueError),
        (None, None, None, 1.0, ValueError),
        (1.0, 1.0, 1.0, 1.0, ValueError),
    ],
)
def test_gamma_process_invalid_parametrization(
    mean: Any,
    variance: Any,
    rate: Any,
    scale: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        _ = TestType(
            mean=mean,
            variance=variance,
            rate=rate,
            scale=scale,
        )


def test_gamma_process_str_repr() -> None:
    instance = TestType(rate=1.1, scale=2.1, t=2.0)

    assert (
        str(instance)
        == "Gamma process with rate = 1.1 and scale = 2.1 on [0, 2.0]"
    )
    assert repr(instance) == "GammaProcess(rate=1.1, scale=2.1, t=2.0)"


def test_gamma_process_sample_shape() -> None:
    instance = TestType(mean=1.1, variance=0.5, t=1.0)
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n + 1,)

    t = np.linspace(0, 1.0, n + 1)
    samples_at = instance.sample_at(times=t)

    assert samples_at.shape == (n + 1,)


@pytest.mark.parametrize(
    "mean, variance, t, n, expected",
    [
        (
            1.1,
            0.5,
            1.0,
            5,
            [0.0, 0.25784847, 0.41911453, 0.80705742, 0.94553971, 1.04164962],
        ),
        (
            1.1,
            0.5,
            2.0,
            5,
            [0.0, 0.54451713, 0.94384016, 1.64596453, 2.77275298, 3.13189101],
        ),
    ],
)
def test_gamma_process_sample_values(
    mean: float,
    variance: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(mean=mean, variance=variance, t=t, rng=rng)

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "mean, variance, t, n, expected",
    [
        (
            1.1,
            0.5,
            1.0,
            5,
            [0.0, 0.25784847, 0.41911453, 0.80705742, 0.94553971, 1.04164962],
        ),
        (
            1.1,
            0.5,
            2.0,
            5,
            [0.0, 0.54451713, 0.94384016, 1.64596453, 2.77275298, 3.13189101],
        ),
    ],
)
def test_gamma_process_sample_at_values(
    mean: float,
    variance: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        mean=mean,
        variance=variance,
        t=t,
        rng=rng,
    )

    times = np.linspace(0, t, n + 1)
    samples_at = instance.sample_at(times=times)
    print(samples_at)
    np.testing.assert_allclose(samples_at, expected, rtol=1e-6)
