import numpy as np
import pytest

from pystochastic.processes.noise import WhiteNoise as TestType


def test_white_noise_str_repr() -> None:
    instance = TestType(t=2.0)

    assert (
        str(instance)
        == "Colored noise generator with exponent 0 on interval [0, 2.0]"
    )
    assert repr(instance) == "ColoredNoise(beta=0, t=2.0)"


def test_white_noise_sample_shape() -> None:
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
            [-0.66301505, -0.13904936, 0.93682859, 0.14933982, -0.284104],
        ),
    ],
)
def test_white_noise_sample_values(
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
