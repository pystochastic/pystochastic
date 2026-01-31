import numpy as np
import pytest

from pystochastic.processes.continuous import (
    FractionalBrownianMotion as TestType,
)


def test_fractional_brownian_motion_str_repr() -> None:
    instance = TestType(hurst=0.7, t=2.0)

    assert (
        str(instance) == "Fractional Brownian motion with "
        "Hurst 0.7 on [0, 2.0]."
    )
    assert repr(instance) == "FractionalBrownianMotion(hurst=0.7, t=2.0)"


@pytest.mark.parametrize(
    "hurst, t, n, expected",
    [
        (
            0.5,
            3.0,
            5,
            [
                0.0,
                -1.579656,
                -1.56551922,
                -2.13299354,
                -1.50615276,
                -1.0449885,
            ],
        ),
        (
            0.75,
            3.0,
            5,
            [
                0.0,
                -1.11693205,
                -2.39810294,
                -3.59909442,
                -4.54417371,
                -4.88149016,
            ],
        ),
    ],
)
def test_fractional_brownian_motion_sample(
    hurst: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        hurst=hurst,
        t=t,
        rng=rng,
    )

    samples = instance.sample(n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)
