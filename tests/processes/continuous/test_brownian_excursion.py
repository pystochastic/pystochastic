import numpy as np
import pytest

from pystochastic.processes.continuous import BrownianExcursion as TestType


def test_brownian_excursion_str_repr() -> None:
    instance = TestType(t=2.0)

    assert str(instance) == "Brownian excursion on [0, 2.0]"
    assert repr(instance) == "BrownianExcursion(t=2.0)"


def test_brownian_excursion_sample_shape() -> None:
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
            [0.0, 0.48257157, 0.86948976, 0.07813982, 0.20696657, 0.0],
        ),
        (
            2.0,
            5,
            [0.0, 0.68245926, 1.22964421, 0.1105064, 0.29269494, 0.0],
        ),
    ],
)
def test_brownian_excursion_sample_values(
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
            [0.0, 0.48257157, 0.86948976, 0.07813982, 0.20696657, 0.0],
        ),
        (
            2.0,
            5,
            [0.0, 0.68245926, 1.22964421, 0.1105064, 0.29269494, 0.0],
        ),
    ],
)
def test_brownian_excursion_sample_at_values(
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
