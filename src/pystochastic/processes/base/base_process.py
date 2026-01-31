from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ...random import generator


class BaseProcess(ABC):
    def __init__(self, rng: Generator | None = None) -> None:
        self.rng = rng

    @property
    def rng(self) -> Generator:
        if self.__rng is None:
            return generator
        return self.__rng

    @rng.setter
    def rng(self, value: Generator | None) -> None:
        if value is None:
            self.__rng = None
        elif isinstance(value, Generator):
            self.__rng = value
        else:
            raise TypeError("rng must be of type `numpy.random.Generator`")

    @abstractmethod
    def sample(self, n: int) -> npt.NDArray[np.float64]:  # pragma: no cover
        pass
