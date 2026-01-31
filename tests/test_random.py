import numpy as np
import pytest

from pystochastic import random


def test_random_default() -> None:
    assert isinstance(random.generator, np.random.Generator)


def test_random_use_generator() -> None:
    random.use_generator()

    assert isinstance(random.generator, np.random.Generator)

    with pytest.raises(TypeError):
        random.use_generator(rng="something")  # type: ignore[arg-type]


def test_random_seed() -> None:
    random.use_generator()

    random.seed(value=42)
    before = random.generator.uniform()

    random.seed(value=42)
    after = random.generator.uniform()

    assert before == after
