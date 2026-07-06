import numpy as np
import pytest

from pystochastic.processes.noise import PinkNoise as TestType


def test_pink_noise_str_repr() -> None:
    instance = TestType(t=2.0)

    assert (
        str(instance)
        == "Colored noise generator with exponent 1 on interval [0, 2.0]"
    )
    assert repr(instance) == "ColoredNoise(beta=1, t=2.0)"


def test_pink_noise_sample_shape() -> None:
    instance = TestType(t=1.0)
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n,)


@pytest.mark.parametrize(
    "t, n, expected",
    [
        (
            2.0,
            5,
            [-0.69704645, -0.09639063, 0.9042075, 0.2335634, -0.34433382],
        ),
    ],
)
def test_pink_noise_sample_values(
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

    assert np.allclose(samples, expected, atol=1e-5)
