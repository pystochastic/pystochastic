from typing import Any

import numpy as np
import pytest

from pystochastic.processes.discrete import MarkovChain as TestType


@pytest.mark.parametrize(
    "transition, error_type",
    [
        (
            "invalid",
            ValueError,
        ),
        (
            -1,
            ValueError,
        ),
        (
            [1, "invalid", 3],
            ValueError,
        ),
        (
            [[0.5, 0.5], [0.5]],
            ValueError,
        ),
        (
            [[0.5, 0.25], [0.5, 0.5]],
            ValueError,
        ),
    ],
)
def test_markov_chain_invalid_transition(
    transition: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(transition=transition)


@pytest.mark.parametrize(
    "transition, initial, error_type",
    [
        (
            [[0.5, 0.5], [0.5, 0.5]],
            "inv",
            ValueError,
        ),
        (
            [[0.5, 0.5], [0.5, 0.5]],
            -1,
            ValueError,
        ),
        (
            [[0.5, 0.5], [0.5, 0.5]],
            [1, "invalid", 3],
            ValueError,
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
        (
            [[0.5, 0.5], [0.5, 0.5]],
            [0.5],
            ValueError,
        ),
        (
            [[0.5, 0.5], [0.5, 0.5]],
            [0.5, 0.25],
            ValueError,
        ),
    ],
)
def test_markov_chain_invalid_weights(
    transition: Any,
    initial: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(transition=transition, initial=initial)


def test_markov_chain_str_repr() -> None:
    instance = TestType(transition=[[0.5, 0.5], [0.5, 0.5]], initial=[0.5, 0.5])

    assert (
        str(instance)
        == "Markov chain with transition matrix = [[0.5 0.5]\n [0.5 0.5]] and initial state"
        " probabilities = [0.5 0.5]"
    )
    assert repr(instance) == "MarkovChain(transition=[[0.5 0.5]\n [0.5 0.5]], initial=[0.5 0.5])"


def test_markov_chain_sample_shape() -> None:
    instance = TestType()
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n,)


@pytest.mark.parametrize(
    "transition, initial, n, expected",
    [
        (
            [[0.5, 0.5], [0.5, 0.5]],
            [0.5, 0.5],
            5,
            [1, 0, 1, 1, 1],
        ),
        (
            [[0.6, 0.4], [0.5, 0.5]],
            [0.5, 0.5],
            5,
            [1, 0, 0, 1, 1],
        ),
    ],
)
def test_markov_chain_sample_values(
    transition: list[list[float]],
    initial: list[float],
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        transition=transition,
        initial=initial,
        rng=rng,
    )

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)
