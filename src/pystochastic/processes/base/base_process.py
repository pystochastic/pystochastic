from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ... import random


class BaseProcess(ABC):
    """Base class to be subclassed to all process classes.
    Contains properties and functions related to random number generation.
    """

    def __init__(
        self,
        *,
        rng: Generator | None = None,
    ) -> None:
        self.rng = rng

    @property
    def rng(self) -> Generator:
        return self.__rng or random.generator

    @rng.setter
    def rng(self, value: Generator | None) -> None:
        if value is None:
            self.__rng = None
        elif isinstance(value, Generator):
            self.__rng = value
        else:
            raise TypeError("rng must be of type `numpy.random.Generator`")

    @abstractmethod
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        """Generate n samples from the process.

        :param int n: the number of samples to generate
        """
