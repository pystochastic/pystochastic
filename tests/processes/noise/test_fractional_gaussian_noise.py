from typing import Any

import numpy as np
import pytest

from pystochastic.processes.noise import FractionalGaussianNoise as TestType


@pytest.mark.parametrize(
    "hurst, error_type",
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
            0.0,
            ValueError,
        ),
        (
            1.0,
            ValueError,
        ),
        (
            -0.5,
            ValueError,
        ),
        (
            1.5,
            ValueError,
        ),
    ],
)
def test_fractional_gaussian_noise_invalid_hurst(
    hurst: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(hurst=hurst)


def test_fractional_gaussian_noise_str_repr() -> None:
    instance = TestType(hurst=0.7, t=2.0)

    assert (
        str(instance) == "Fractional Gaussian noise with "
        "Hurst 0.7 on [0, 2.0]."
    )
    assert repr(instance) == "FractionalGaussianNoise(hurst=0.7, t=2.0)"


@pytest.mark.parametrize(
    "hurst, t, n, expected",
    [
        (
            0.5,
            3.0,
            5,
            [-1.579656, 0.01413678, -0.56747432, 0.62684077, 0.46116426],
        ),
        (
            0.75,
            3.0,
            5,
            [-1.11693205, -1.2811709, -1.20099148, -0.94507929, -0.33731645],
        ),
    ],
)
def test_fractional_gaussian_noise_sample(
    hurst: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        hurst=hurst,
        t=t,
        rng=rng,
    )

    samples = instance.sample(n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "hurst, t, n, expected",
    [
        (
            0.5,
            3.0,
            5,
            [-1.579656, 0.01413678, -0.56747432, 0.62684077, 0.46116426],
        ),
        (
            0.75,
            3.0,
            5,
            [-1.11693205, -1.2811709, -1.20099148, -0.94507929, -0.33731645],
        ),
    ],
)
def test_fractional_gaussian_noise_sample_davies_harte(
    hurst: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        hurst=hurst,
        t=t,
        rng=rng,
    )

    samples = instance.sample_davies_harte(n)
    print(samples)

    np.testing.assert_allclose(samples, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "hurst, t, n, expected",
    [
        (
            0.5,
            3.0,
            5,
            [-1.579656, 0.01413678, -0.56747432, 0.62684077, 0.46116426],
        ),
        (
            0.75,
            3.0,
            5,
            [-1.39027378, -0.56454585, -0.82213465, 0.03581026, 0.18580369],
        ),
    ],
)
def test_fractional_gaussian_noise_sample_hosking(
    hurst: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        hurst=hurst,
        t=t,
        rng=rng,
    )

    samples = instance.sample_hosking(n)
    print(samples)

    np.testing.assert_allclose(samples, expected, rtol=1e-6)
