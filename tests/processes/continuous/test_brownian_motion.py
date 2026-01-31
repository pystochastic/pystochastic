from typing import Any

import numpy as np
import pytest

from pystochastic.processes.continuous import BrownianMotion as TestType


@pytest.mark.parametrize(
    "drift, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
    ],
)
def test_brownian_motion_invalid_drift(
    drift: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(drift=drift)


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
def test_brownian_motion_invalid_scale(
    scale: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(scale=scale)


def test_brownian_motion_str_repr() -> None:
    instance = TestType(drift=0.5, scale=2.0, t=1.0)

    assert (
        str(instance) == "Brownian motion with drift 0.5 and "
        "scale 2.0 on interval [0, 1.0]."
    )

    assert repr(instance) == "BrownianMotion(drift=0.5, scale=2.0, t=1.0)"

    instance_standard = TestType()

    assert (
        str(instance_standard)
        == "Standard Brownian motion on interval [0, 1.0]."
    )
    assert (
        repr(instance_standard) == "BrownianMotion(drift=0.0, scale=1.0, "
        "t=1.0)"
    )


def test_brownian_motion_sample_shape() -> None:
    instance = TestType(t=1.0)
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n + 1,)

    t = np.linspace(0, 1.0, n + 1)
    samples_at = instance.sample_at(times=t)

    assert samples_at.shape == (n + 1,)


@pytest.mark.parametrize(
    "drift, scale, t, n, expected",
    [
        (
            0.5,
            2.0,
            1.0,
            5,
            [
                0.0,
                -1.72402963,
                -1.60770588,
                -2.16296879,
                -1.33915541,
                -0.70664878,
            ],
        ),
        (
            0.0,
            1.0,
            1.0,
            5,
            [
                0.0,
                -0.91201481,
                -0.90385294,
                -1.23148439,
                -0.8695777,
                -0.60332439,
            ],
        ),
    ],
)
def test_brownian_motion_sample_values(
    drift: float,
    scale: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        drift=drift,
        scale=scale,
        t=t,
        rng=rng,
    )

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "drift, scale, t, n, expected",
    [
        (
            0.5,
            2.0,
            1.0,
            5,
            [
                0.0,
                -1.72402963,
                -1.60770588,
                -2.16296879,
                -1.33915541,
                -0.70664878,
            ],
        ),
        (
            0.0,
            1.0,
            1.0,
            5,
            [
                0.0,
                -0.91201481,
                -0.90385294,
                -1.23148439,
                -0.8695777,
                -0.60332439,
            ],
        ),
    ],
)
def test_brownian_motion_sample_at_values(
    drift: float,
    scale: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        drift=drift,
        scale=scale,
        t=t,
        rng=rng,
    )

    times = np.linspace(0, t, n + 1)
    samples_at = instance.sample_at(times=times)
    print(samples_at)
    np.testing.assert_allclose(samples_at, expected, rtol=1e-6)
