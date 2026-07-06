import numpy as np
import pytest

from pystochastic.processes.continuous import BrownianMeander as TestType


def test_brownian_excursion_str_repr() -> None:
    instance = TestType(t=2.0)

    assert str(instance) == "Brownian meander on [0, 2.0]"
    assert repr(instance) == "BrownianMeander(t=2.0)"


def test_brownian_meander_sample_shape() -> None:
    instance = TestType(t=1.0)
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n + 1,)

    t = np.linspace(0, 1.0, n + 1)
    samples_at = instance.sample_at(times=t)

    assert samples_at.shape == (n + 1,)


@pytest.mark.parametrize(
    "t, n, expected",
    [
        (
            1.0,
            5,
            [0.0, 0.33817703, 0.43730619, 1.06573627, 1.60145602, 1.94307116],
        ),
        (
            2.0,
            5,
            [0.0, 0.47825454, 0.61844435, 1.50717869, 2.26480083, 2.74791758],
        ),
    ],
)
def test_brownian_meander_sample_values(
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(t=t, rng=rng)

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "t, n, expected",
    [
        (
            1.0,
            5,
            [0.0, 0.33817703, 0.43730619, 1.06573627, 1.60145602, 1.94307116],
        ),
        (
            2.0,
            5,
            [0.0, 0.47825454, 0.61844435, 1.50717869, 2.26480083, 2.74791758],
        ),
    ],
)
def test_brownian_meander_sample_at_values(
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
    samples_at = instance.sample_at(times=times)
    print(samples_at)
    np.testing.assert_allclose(samples_at, expected, rtol=1e-6)
