from typing import Any

import numpy as np
import pytest

from pystochastic.processes.continuous import (
    GeometricBrownianMotion as TestType,
)


@pytest.mark.parametrize(
    "drift, volatility, error_type",
    [
        (1.0, None, TypeError),
        (None, 1.0, TypeError),
        (1.0, -1.0, ValueError),
        (1.0, 0.0, ValueError),
    ],
)
def test_geometric_brownian_motion_invalid_parametrization(
    drift: Any,
    volatility: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        _ = TestType(drift=drift, volatility=volatility)


def test_geometric_brownian_motion_str_repr() -> None:
    instance = TestType(drift=1.1, volatility=2.1, t=2.0)

    assert (
        str(instance)
        == "Geometric Brownian motion with drift 1.1 and volatility"
        " 2.1 on [0, 2.0]."
    )
    assert (
        repr(instance)
        == "GeometricBrownianMotion(drift=1.1, volatility=2.1, t=2.0)"
    )


def test_geometric_brownian_motion_sample_shape() -> None:
    instance = TestType(drift=1.1, volatility=0.5, t=1.0)
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n + 1,)

    t = np.linspace(0, 1.0, n + 1)
    samples_at = instance.sample_at(times=t)

    assert samples_at.shape == (n + 1,)


@pytest.mark.parametrize(
    "drift, volatility, t, n, expected",
    [
        (
            1.1,
            0.5,
            1.0,
            5,
            [1.0, 0.88160832, 1.2313037, 1.45391193, 2.42349605, 3.85101906],
        ),
        (
            1.1,
            0.5,
            2.0,
            5,
            [1.0, 0.7298678, 1.02109896, 1.12660539, 2.02408088, 3.39867358],
        ),
    ],
)
def test_geometric_brownian_motion_sample_values(
    drift: float,
    volatility: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        drift=drift,
        volatility=volatility,
        t=t,
        rng=rng,
    )

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "drift, volatility, t, n, expected",
    [
        (
            1.1,
            0.5,
            1.0,
            5,
            [1.0, 0.85984133, 1.17125231, 1.34885732, 2.19286991, 3.39851239],
        ),
        (
            1.1,
            0.5,
            2.0,
            5,
            [1.0, 0.96570984, 1.78760986, 2.60963153, 6.20351278, 13.78230098],
        ),
    ],
)
def test_geometric_brownian_motion_sample_at_values(
    drift: float,
    volatility: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        drift=drift,
        volatility=volatility,
        t=t,
        rng=rng,
    )

    times = np.linspace(0, t, n + 1)
    samples_at = instance.sample_at(times=times)
    print(samples_at)
    np.testing.assert_allclose(samples_at, expected, rtol=1e-6)
