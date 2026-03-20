from typing import Callable

import pytest

from pystochastic import random
from pystochastic.processes.discrete import DirichletProcess


def test_dirichlet_process_str_repr(
    base: Callable[[], float] | None,
    alpha: float,
) -> None:
    if (not callable(base) and base is not None) or alpha <= 0:
        with pytest.raises(ValueError):
            _ = DirichletProcess(base, alpha)
        return

    instance = DirichletProcess(base, alpha)

    if base is None:
        assert instance.base == random.generator.uniform

    assert isinstance(repr(instance), str)
    assert isinstance(str(instance), str)


def test_dirichlet_process_sample(
    base: Callable[[], float] | None,
    alpha: float,
    n: int,
) -> None:
    if (not callable(base) and base is not None) or alpha <= 0:
        with pytest.raises(ValueError):
            _ = DirichletProcess(base, alpha)
        return

    instance = DirichletProcess(base, alpha)

    s = instance.sample(n)
    assert len(s) == n
