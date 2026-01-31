from typing import Any

import numpy as np
import pytest

from pystochastic.processes.continuous import BesselProcess as TestType


@pytest.mark.parametrize(
    "dim, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
        (
            -1,
            ValueError,
        ),
    ],
)
def test_bessel_process_invalid_dim(
    dim: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(dim=dim)


def test_bessel_process_str_repr() -> None:
    instance = TestType(dim=1, t=2.0)

    assert str(instance) == "Bessel process of 1 Wiener processes on [0, 2.0]"
    assert repr(instance) == "BesselProcess(dim=1, t=2.0)"


def test_bessel_process_sample_shape() -> None:
    instance = TestType(t=1.0)
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n + 1,)

    t = np.linspace(0, 1.0, n + 1)
    samples_at = instance.sample_at(times=t)

    assert samples_at.shape == (n + 1,)


@pytest.mark.parametrize(
    "dim, t, n, expected",
    [
        (
            1,
            1.0,
            5,
            [0.0, 0.91201481, 0.90385294, 1.23148439, 0.8695777, 0.60332439],
        ),
        (
            2,
            1.0,
            5,
            [0.0, 0.92018353, 0.91046622, 1.23981509, 0.87021971, 0.60433644],
        ),
    ],
)
def test_bessel_process_sample_values(
    dim: int,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        dim=dim,
        t=t,
        rng=rng,
    )

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)
