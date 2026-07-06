from typing import Any

import numpy as np
import pytest

from pystochastic.processes.discrete import BernoulliProcess as TestType


@pytest.mark.parametrize(
    "p, error_type",
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
            1.5,
            ValueError,
        ),
    ],
)
def test_bernoulli_process_invalid_p(
    p: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(p=p)


def test_bernoulli_process_str_repr() -> None:
    instance = TestType(p=0.5)

    assert str(instance) == "Bernoulli process with p=0.5."
    assert repr(instance) == "BernoulliProcess(0.5)"


def test_bernoulli_process_sample_shape() -> None:
    instance = TestType()
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n,)


@pytest.mark.parametrize(
    "p, n, expected",
    [
        (
            0.5,
            5,
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ),
        (
            0.6,
            5,
            [0.0, 1.0, 1.0, 0.0, 0.0],
        ),
    ],
)
def test_bernoulli_process_sample_values(
    p: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        p=p,
        rng=rng,
    )

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)
