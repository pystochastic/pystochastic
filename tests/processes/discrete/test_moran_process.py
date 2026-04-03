from typing import Any

import numpy as np
import pytest

from pystochastic.processes.discrete import MoranProcess as TestType


@pytest.mark.parametrize(
    "maximum, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
        (
            -1,
            ValueError,
        ),
        (
            2,
            ValueError,
        ),
    ],
)
def test_moran_process_invalid_maximum(
    maximum: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(maximum=maximum)


def test_moran_process_str_repr() -> None:
    instance = TestType(maximum=5)

    assert str(instance) == "Moran process with 5 states"
    assert repr(instance) == "MoranProcess(maximum=5)"


def test_moran_process_sample_shape() -> None:
    instance = TestType(maximum=500)
    n = 100
    samples = instance.sample(n=n)

    assert samples.shape == (n,)


@pytest.mark.parametrize(
    "maximum, start, n, expected",
    [
        (
            10,
            3,
            5,
            [3, 3, 2, 2, 2],
        ),
        (
            20,
            6,
            5,
            [6, 6, 5, 5, 5],
        ),
    ],
)
def test_moran_process_sample_values(
    maximum: int,
    start: int,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        maximum=maximum,
        rng=rng,
    )

    samples = instance.sample_with_start(n=n, start=start)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)
