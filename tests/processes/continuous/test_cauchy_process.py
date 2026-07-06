import numpy as np
import pytest

from pystochastic.processes.continuous import CauchyProcess as TestType


def test_cauchy_process_str_repr() -> None:
    instance = TestType(t=2.0)

    assert str(instance) == "Cauchy process on [0, 2.0]"
    assert repr(instance) == "CauchyProcess(t=2.0)"


def test_cauchy_process_sample_shape() -> None:
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
            [0.0, 0.10316847, 0.05513344, 0.2067938, 0.08257761, 0.0844287],
        ),
        (
            2.0,
            5,
            [0.0, 0.20633694, 0.11026688, 0.41358761, 0.16515522, 0.16885739],
        ),
    ],
)
def test_cauchy_process_sample_values(
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(t=t, rng=rng)

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)
