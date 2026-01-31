import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ...utils import generate_times
from ...utils.validation import check_positive_integer, check_positive_number
from .base_process import BaseProcess


class BaseTimeProcess(BaseProcess):
    """Base class to be subclassed to most process classes.

    Contains properties and functions related to times and continuous-time
    processes.
    """

    def __init__(
        self,
        t: float = 1,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)

        self.t = t
        self.__n: int | None = None
        self.__times: npt.NDArray[np.float64] | None = None

    @property
    def t(self) -> float:
        """End time of the process."""
        return self.__t

    @t.setter
    def t(self, value: float) -> None:
        check_positive_number(value, "Time end")
        self.__t = float(value)

    def _set_times(self, n: int) -> None:
        if self.__n != n:
            check_positive_integer(n)
            self.__n = n
            self.__times = generate_times(self.t, n)

    def times(self, n: int) -> npt.NDArray[np.float64]:
        """Generate times associated with n increments on [0, t].

        :param int n: the number of increments
        """
        self._set_times(n)

        assert self.__times is not None

        return self.__times
