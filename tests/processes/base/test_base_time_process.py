from typing import Any, override

import numpy as np
import numpy.typing as npt
import pytest

from pystochastic import random
from pystochastic.processes.base import BaseTimeProcess


class SubBaseTimeProcess(BaseTimeProcess):

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return np.zeros(n)

    @override
    def sample_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return np.zeros(len(times))


def test_base_time_process_abstract() -> None:
    with pytest.raises(TypeError):
        # pylint: disable=abstract-class-instantiated
        _ = BaseTimeProcess()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "t, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
        (
            0,
            ValueError,
        ),
        (
            -1,
            ValueError,
        ),
    ],
)
def test_base_time_process_constructor_invalid_t(
    t: Any,
    error_type: type,
) -> None:

    with pytest.raises(error_type):
        _ = SubBaseTimeProcess(t=t)


def test_base_time_process_constructor() -> None:
    instance = SubBaseTimeProcess()

    assert instance.rng == random.generator
    assert instance.t == 1.0

    rng = np.random.default_rng(seed=324)
    instance = SubBaseTimeProcess(rng=rng)

    assert instance.rng == rng
    assert instance.t == 1.0


@pytest.mark.parametrize(
    "end_t, n",
    [
        (
            2.0,
            0,
        ),
        (
            2.0,
            -1,
        ),
        # (
        #     0.0,
        #     4,
        # ),
        # (
        #     -2.0,
        #     4,
        # ),
    ],
)
def test_base_time_process_set_times_with_t_invalid(
    end_t: float,
    n: int,
) -> None:
    instance = SubBaseTimeProcess()

    with pytest.raises(ValueError):
        _ = instance.set_times_with_t(end_t=end_t, n=n)


def test_base_time_process_set_times_with_t() -> None:
    instance = SubBaseTimeProcess()

    result = instance.set_times_with_t(end_t=3.0, n=4)

    assert result is True
    assert instance.n == 4
    assert len(instance.times) == 5  # n increments -> n+1 times
    assert instance.times[0] == 0.0
    assert instance.times[-1] == 3.0


@pytest.mark.parametrize(
    "n",
    [
        0,
        -3,
    ],
)
def test_base_time_process_set_times_invalid(n: int) -> None:
    instance = SubBaseTimeProcess(t=2.0)

    with pytest.raises(ValueError):
        _ = instance.set_times(n=n)


def test_base_time_process_set_times() -> None:
    instance = SubBaseTimeProcess(t=2.0)

    result = instance.set_times(n=5)

    assert result is True
    assert instance.n == 5
    assert len(instance.times) == 6  # n increments -> n+1 times
    assert instance.times[0] == 0.0
    assert instance.times[-1] == 2.0
