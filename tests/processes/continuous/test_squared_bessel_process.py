import numpy as np
import pytest

from pystochastic.processes.continuous import SquaredBesselProcess as TestType


def test_squared_bessel_process_str_repr() -> None:
    instance = TestType(dim=1, t=2.0)

    assert (
        str(instance)
        == "Squared Bessel process of 1 Wiener processes on [0, 2.0]"
    )
    assert repr(instance) == "SquaredBesselProcess(dim=1, t=2.0)"


@pytest.mark.parametrize(
    "dim, t, n, expected",
    [
        (
            1,
            1.0,
            5,
            [0.0, 0.83177102, 0.81695014, 1.51655381, 0.75616538, 0.36400032],
        ),
        (
            2,
            1.0,
            5,
            [0.0, 0.84673772, 0.82894874, 1.53714146, 0.75728235, 0.36522253],
        ),
    ],
)
def test_squared_bessel_process_sample_values(
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
