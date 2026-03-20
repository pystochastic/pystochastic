from typing import Any

import numpy as np
import pytest

from pystochastic.processes.continuous import VarianceGammaProcess as TestType


@pytest.mark.parametrize(
    "drift, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
    ],
)
def test_variance_gamma_process_invalid_drift(
    drift: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(drift=drift)


@pytest.mark.parametrize(
    "variance, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
        (
            0.0,
            ValueError,
        ),
        (
            -0.5,
            ValueError,
        ),
    ],
)
def test_variance_gamma_process_invalid_variance(
    variance: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(variance=variance)


@pytest.mark.parametrize(
    "scale, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
        (
            0.0,
            ValueError,
        ),
        (
            -0.5,
            ValueError,
        ),
    ],
)
def test_variance_gamma_process_invalid_scale(
    scale: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(scale=scale)


def test_variance_gamma_process_str_repr() -> None:
    instance = TestType(drift=0.5, scale=2.0, t=1.0)

    assert (
        str(instance)
        == "Variance Gamma process on interval [0, 1.0] with drift 0.5, variance 1.0, and scale 2.0."
    )

    assert repr(instance) == "VarianceGammaProcess(drift=0.5, variance=1.0, scale=2.0, t=1.0)"


def test_variance_gamma_process_sample_shape() -> None:
    instance = TestType(t=1.0)
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n + 1,)

    t = np.linspace(0, 1.0, n + 1)
    samples_at = instance.sample_at(times=t)

    assert samples_at.shape == (n + 1,)


@pytest.mark.parametrize(
    "drift, variance, scale, t, n, expected",
    [
        (
            0.5,
            1.0,
            2.0,
            1.0,
            5,
            [0.0, -0.02246366, 0.0562891, -0.15816879, -0.26618616, 0.06006846],
        ),
        (
            0.0,
            1.1,
            1.0,
            1.0,
            5,
            [0.0, -0.0534869, -0.03477381, -0.36238551, -0.42393261, -0.28722455],
        ),
    ],
)
def test_variance_gamma_process_sample_values(
    drift: float,
    variance: float,
    scale: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        drift=drift,
        variance=variance,
        scale=scale,
        t=t,
        rng=rng,
    )

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "drift, variance, scale, t, n, expected",
    [
        (
            0.5,
            1.0,
            2.0,
            1.0,
            5,
            [0.0, -0.02246366, 0.0562891, -0.15816879, -0.26618616, 0.06006846],
        ),
        (
            0.0,
            1.1,
            1.0,
            1.0,
            5,
            [0.0, -0.0534869, -0.03477381, -0.36238551, -0.42393261, -0.28722455],
        ),
    ],
)
def test_variance_gamma_process_sample_at_values(
    drift: float,
    variance: float,
    scale: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        drift=drift,
        variance=variance,
        scale=scale,
        t=t,
        rng=rng,
    )

    times = np.linspace(0, t, n + 1)
    samples_at = instance.sample_at(times=times)
    print(samples_at)
    np.testing.assert_allclose(samples_at, expected, rtol=1e-6)
