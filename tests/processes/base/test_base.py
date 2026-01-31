import numpy as np
import pytest

from pystochastic import random
from pystochastic.processes.base import (
    BaseProcess,
    BaseSequenceProcess,
    BaseTimeProcess,
)


def test_base_process(end, n):
    with pytest.raises(TypeError):
        _ = BaseProcess()


def test_base_process_rng():
    class SubBaseProcess(BaseProcess):
        def __init__(self, rng=None):
            super().__init__(rng=rng)

        def sample(self, n):
            return np.zeros(n)

    sub = SubBaseProcess(rng=None)
    assert sub.rng == random.generator

    generator = np.random.default_rng()
    sub = SubBaseProcess(rng=generator)
    assert sub.rng == generator


def test_base_sequence_process(end, n):
    with pytest.raises(TypeError):
        _ = BaseSequenceProcess()


def test_base_time_process(end, n):
    with pytest.raises(TypeError):
        _ = BaseTimeProcess()
