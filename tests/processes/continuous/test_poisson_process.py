from typing import Any

import numpy as np
import pytest

from pystochastic.processes.continuous import PoissonProcess as TestType


@pytest.mark.parametrize(
    "rate, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
        (
            -1.0,
            ValueError,
        ),
        (
            -0.0001,
            ValueError,
        ),
    ],
)
def test_poisson_process_invalid_rate(
    rate: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(rate=rate)


def test_poisson_process_str_repr() -> None:
    instance = TestType(rate=1.0)

    assert str(instance) == "Poisson process with rate 1.0."
    assert repr(instance) == "PoissonProcess(rate=1.0)"


def test_poisson_process_sample_shape() -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(rate=1.0, rng=rng)
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n + 1,)

    samples_at = instance.sample_with_length(length=float(n))

    assert samples_at.shape == (1005,)


@pytest.mark.parametrize(
    "rate, n, expected",
    [
        (
            0.5,
            5,
            [0.0, 3.775526, 4.516659, 7.135515, 8.044721, 11.090316],
        ),
        (
            5.0,
            5,
            [0.0, 0.3775525515815201, 0.451666, 0.713552, 0.804472, 1.109032],
        ),
    ],
)
def test_poisson_process_sample_values(
    rate: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        rate=rate,
        rng=rng,
    )

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "rate, length, expected",
    [
        (
            0.5,
            5.0,
            [0.0, 3.775526, 4.516659, 7.135515],
        ),
        (
            1.0,
            5.0,
            [0.0, 1.887763, 2.25833, 3.567758, 4.02236, 5.545158],
        ),
    ],
)
def test_poisson_process_sample_with_length(
    rate: float,
    length: float,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        rate=rate,
        rng=rng,
    )

    samples = instance.sample_with_length(length=length)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)
