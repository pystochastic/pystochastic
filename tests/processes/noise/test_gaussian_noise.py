import numpy as np
import pytest

from pystochastic.processes.noise import GaussianNoise as TestType


def test_gaussian_noise_str_repr() -> None:
    instance = TestType(t=2.0)

    assert str(instance) == "Gaussian noise generator on interval [0, 2.0]"
    assert repr(instance) == "GaussianNoise(t=2.0)"


def test_gaussian_noise_sample_shape() -> None:
    instance = TestType(t=1.0)
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n,)

    t = np.linspace(0, 1.0, n + 1)
    samples_at = instance.sample_at(times=t)

    assert samples_at.shape == (n,)


@pytest.mark.parametrize(
    "t, n, expected",
    [
        (
            2.0,
            5,
            [-1.28978372, 0.01154263, -0.46334084, 0.51181335, 0.37653904],
        ),
    ],
)
def test_gaussian_noise_sample_values(
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        t=t,
        rng=rng,
    )

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "t, n, expected",
    [
        (
            3.0,
            5,
            [-1.579656, 0.01413678, -0.56747432, 0.62684077, 0.46116426],
        ),
    ],
)
def test_gaussian_noise_sample_at_values(
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        t=t,
        rng=rng,
    )

    times = np.linspace(0, t, n + 1)
    samples = instance.sample_at(times=times)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)
