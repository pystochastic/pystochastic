import numpy as np
import pytest

from pystochastic.processes.noise import RedNoise as TestType


def test_red_noise_str_repr() -> None:
    instance = TestType(t=2.0)

    assert (
        str(instance)
        == "Colored noise generator with exponent 2 on interval [0, 2.0]"
    )
    assert repr(instance) == "ColoredNoise(beta=2, t=2.0)"


def test_red_noise_sample_shape() -> None:
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
            [-0.71658961, -0.06305761, 0.87115416, 0.29551432, -0.38702126],
        ),
    ],
)
def test_red_noise_sample_values(
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
