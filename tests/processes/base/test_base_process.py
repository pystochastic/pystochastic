from typing import Any, override

import numpy as np
import numpy.typing as npt
import pytest

from pystochastic import random
from pystochastic.processes.base import BaseProcess


class SubBaseProcess(BaseProcess):

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return np.zeros(n)


def test_base_process_abstract() -> None:
    with pytest.raises(TypeError):
        # pylint: disable=abstract-class-instantiated
        _ = BaseProcess()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "rng, error_type",
    [
        (
            "not a generator",
            TypeError,
        ),
    ],
)
def test_base_process_constructor_invalid_generator(
    rng: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        _ = SubBaseProcess(rng=rng)


def test_base_process_constructor() -> None:
    instance = SubBaseProcess()

    assert instance.rng == random.generator

    rng = np.random.default_rng(seed=123)
    instance = SubBaseProcess(rng=rng)

    assert instance.rng == rng
