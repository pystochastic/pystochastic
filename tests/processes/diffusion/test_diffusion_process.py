from typing import Any

import numpy as np
import pytest

from pystochastic.processes.diffusion import DiffusionProcess as TestType


@pytest.mark.parametrize(
    "speed, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
        (
            lambda x, y: x + y,
            ValueError,
        ),
    ],
)
def test_diffusion_process_invalid_speed(
    speed: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        TestType(speed=speed)


def test_diffusion_process_str_repr() -> None:
    instance = TestType(speed=1.0, mean=0.0, vol=1.0, t=2.0)

    assert "Diffusion process with speed=" in str(instance)
    assert "DiffusionProcess(speed=" in repr(instance)


def test_diffusion_process_sample_shape() -> None:
    instance = TestType(
        speed=1.0,
        mean=0.0,
        vol=1.0,
        volexp=0.0,
        t=1.0,
    )
    n = 1000
    samples = instance.sample(n=n)

    assert samples.shape == (n + 1,)


@pytest.mark.parametrize(
    "speed,mean,vol,volexp, t, n, expected",
    [
        (
            1,
            1.0,
            1.0,
            0.0,
            1.0,
            5,
            [1.0, 0.08798519, 0.27855002, 0.09520856, 0.63807354, 0.97671214],
        ),
        (
            2,
            1.0,
            5,
            0.0,
            1.0,
            5,
            [1.0, -3.56007407, -1.69523508, -2.25529831, 0.85635447, 2.24507924],
        ),
    ],
)
def test_diffusion_process_sample_values(
    speed: float,
    mean: float,
    vol: float,
    volexp: float,
    t: float,
    n: int,
    expected: list[float],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = TestType(
        speed=speed,
        mean=mean,
        vol=vol,
        volexp=volexp,
        t=t,
        rng=rng,
    )

    samples = instance.sample(n=n)
    print(samples)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)
