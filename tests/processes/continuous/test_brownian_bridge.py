from typing import Any

import numpy as np
import pytest

from pystochastic.processes.continuous import BrownianBridge as TestType


@pytest.mark.parametrize(
    "b, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
    ],
)
def test_brownian_bridge_invalid_b(
    b: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(b=b)


def test_brownian_bridge_str_repr() -> None:
    instance = TestType(b=1.0, t=2.0)

    assert str(instance) == "Brownian bridge from 0 to 1.0 on [0, 2.0]"
    assert repr(instance) == "BrownianBridge(b=1.0, t=2.0)"


def test_brownian_bridge_sample_shape() -> None:
    instance = TestType(t=1.0)
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n + 1,)

    t = np.linspace(0, 1.0, n + 1)
    samples_at = instance.sample_at(times=t)

    assert samples_at.shape == (n + 1,)


@pytest.mark.parametrize(
    "b, t, n, expected",
    [
        (
            0.5,
            1.0,
            5,
            [0.0, -0.69134994, -0.46252318, -0.56948976, 0.01308181, 0.5],
        ),
        (
            0.0,
            1.0,
            5,
            [0.0, -0.79134994, -0.66252318, -0.86948976, -0.38691819, 0.0],
        ),
    ],
)
def test_brownian_motion_sample_values(
    b: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        b=b,
        t=t,
        rng=rng,
    )

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "b, t, n, expected",
    [
        (
            0.5,
            1.0,
            5,
            [0.0, -0.69134994, -0.46252318, -0.56948976, 0.01308181, 0.5],
        ),
        (
            0.0,
            1.0,
            5,
            [0.0, -0.79134994, -0.66252318, -0.86948976, -0.38691819, 0.0],
        ),
    ],
)
def test_brownian_motion_sample_at_values(
    b: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        b=b,
        t=t,
        rng=rng,
    )

    times = np.linspace(0, t, n + 1)
    samples_at = instance.sample_at(times=times)
    print(samples_at)
    np.testing.assert_allclose(samples_at, expected, rtol=1e-6)
