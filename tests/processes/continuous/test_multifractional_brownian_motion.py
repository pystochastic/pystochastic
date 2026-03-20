from typing import Any

import numpy as np
import pytest

from pystochastic.processes.continuous import MultifractionalBrownianMotion as TestType


@pytest.mark.parametrize(
    "hurst, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
        (
            -1.0,
            TypeError,
        ),
    ],
)
def test_multifractional_brownian_motion_invalid_hurst(
    hurst: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(hurst=hurst)


def test_multifractional_brownian_motion_str_repr() -> None:
    instance = TestType(hurst=lambda t: 0.5, t=1.0)

    assert (
        str(instance) == "Multifractional Brownian motion with Hurst function <lambda> on [0, 1.0]."
    )
    assert repr(instance) == "MultifractionalBrownianMotion(hurst=<lambda>, t=1.0)"


def test_multifractional_brownian_motion_sample_shape() -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(hurst=lambda t: 0.5, t=1.0, rng=rng)
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n + 1,)


@pytest.mark.parametrize(
    "hurst_func, t, n, expected",
    [
        (
            lambda t: 0.5,
            1.0,
            5,
            [0.0, -0.91201481, -0.90385294, -1.23148439, -0.8695777, -0.60332439],
        ),
        (
            lambda t: 0.5 + 0.4 * t,
            1.0,
            5,
            [0.0, -0.775726, -0.80142311, -1.04402506, -0.96164799, -0.88792645],
        ),
    ],
)
def test_multifractional_brownian_motion_sample_values(
    hurst_func: Any,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        hurst=hurst_func,
        t=t,
        rng=rng,
    )

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)
