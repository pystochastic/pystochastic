from typing import Any, Callable

import numpy as np
import pytest

from pystochastic.processes.discrete import DirichletProcess as TestType


@pytest.mark.parametrize(
    "base, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
    ],
)
def test_dirichlet_process_invalid_base(
    base: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(base=base)


@pytest.mark.parametrize(
    "alpha, error_type",
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
            0,
            ValueError,
        ),
    ],
)
def test_dirichlet_process_invalid_alpha(
    alpha: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(alpha=alpha)


def test_dirichlet_process_str_repr() -> None:
    instance = TestType(
        base=lambda: 0.5,
        alpha=1,
    )

    assert "Dirichlet process with alpha=1" in str(instance)
    assert "DirichletProcess(alpha=1" in repr(instance)


def test_dirichlet_process_sample_shape() -> None:
    instance = TestType(alpha=1)
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n,)


@pytest.mark.parametrize(
    "base, alpha, n, expected",
    [
        (
            lambda: 0.5,
            0.8,
            5,
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ),
    ],
)
def test_dirichlet_process_sample_values(
    base: Callable[[], float],
    alpha: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        base=base,
        alpha=alpha,
        rng=rng,
    )

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)
