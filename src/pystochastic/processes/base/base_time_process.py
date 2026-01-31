import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ...utils import generate_times
from ...utils.validation import check_positive_integer, check_positive_number
from .base_process import BaseProcess


class BaseTimeProcess(BaseProcess):
    """Base class to be subclassed to all time process classes.

    Contains properties and functions related to times and continuous-time
    processes.
    """

    def __init__(
        self,
        *,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)

        self.t = t

        self.__n: int | None = None
        self.__times: npt.NDArray[np.float64] | None = None

    @property
    def t(self) -> float:
        """End time of the process."""
        return self._t

    @t.setter
    def t(self, value: float) -> None:
        check_positive_number(value=value, name="Time end")

        self._t = float(value)

    @property
    def n(self) -> int:
        """Number of increments for which times have been generated."""
        assert self.__n is not None
        return self.__n

    @property
    def times(self) -> npt.NDArray[np.float64]:
        """Times associated with the increments."""
        assert self.__times is not None
        return self.__times

    def set_times_with_t(self, end_t: float, n: int) -> bool:
        """Generate times associated with n increments on [0, end_t].

        :param float end_t: the right hand endpoint of the time
            interval [0, end_t] for the process
        :param int n: the number of increments
        """
        check_positive_integer(n=n, name="Number of increments")
        # check_positive_number(end_t, "End time")

        if (
            self.__n == n
            and self.__times is not None
            and self.__times[-1] == end_t
        ):
            return False

        self.__n = n
        self.__times = generate_times(end=end_t, n=n)

        return True

    def set_times(self, n: int) -> bool:
        """Generate times associated with n increments on [0, t].

        :param int n: the number of increments
        """
        return self.set_times_with_t(self.t, n)

    def sample_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate samples at specified times from zero.

        :param times: a vector of increasing time values for which to generate
            samples.
        """
        return self.sample(n=len(times))
