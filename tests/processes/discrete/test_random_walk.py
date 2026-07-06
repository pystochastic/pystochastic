from typing import Any

import numpy as np
import pytest

from pystochastic.processes.discrete import RandomWalk as TestType


@pytest.mark.parametrize(
    "steps, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
        (
            -1,
            TypeError,
        ),
        (
            [1, "invalid", 3],
            TypeError,
        ),
    ],
)
def test_random_walk_invalid_steps(
    steps: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(steps=steps)


@pytest.mark.parametrize(
    "steps, weights, error_type",
    [
        (
            [-1, 0, 1],
            "inv",
            TypeError,
        ),
        (
            [-1, 0, 1],
            -1,
            TypeError,
        ),
        (
            [-1, 0, 1],
            [1, "invalid", 3],
            TypeError,
        ),
        (
            [-1, 0, 1],
            [1, -1, 3],
            ValueError,
        ),
        (
            [-1, 0, 1],
            [1, 2],
            ValueError,
        ),
    ],
)
def test_random_walk_invalid_weights(
    steps: Any,
    weights: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(steps=steps, weights=weights)


def test_random_walk_str_repr() -> None:
    instance = TestType(steps=[-1, 0, 1])

    assert str(instance) == "Random walk steps = [-1.  0.  1.] and weights = [1. 1. 1.]"
    assert repr(instance) == "RandomWalk(steps=[-1.  0.  1.], weights=[1. 1. 1.])"


def test_random_walk_sample_shape() -> None:
    instance = TestType()
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n + 1,)


@pytest.mark.parametrize(
    "steps, weights, n, expected",
    [
        (
            [-1, 0, 1],
            None,
            5,
            [0, 1, 0, 0, 1, 2],
        ),
        (
            [-1, 0, 1],
            [1, 2, 3],
            5,
            [0, 1, 0, 1, 2, 3],
        ),
    ],
)
def test_random_walk_sample_values(
    steps: list[float | int],
    weights: list[float | int] | None,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        steps=steps,
        weights=weights,
        rng=rng,
    )

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)
