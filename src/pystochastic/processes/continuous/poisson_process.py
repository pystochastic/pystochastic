from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ...utils.validation import (
    check_nonnegative_number,
    check_positive_integer,
    check_positive_number,
)
from ..base import BaseProcess


class PoissonProcess(BaseProcess):
    r"""Poisson process.

    .. image:: _static/poisson_process.png
        :scale: 50%

    A Poisson process with rate :math:`\lambda` is a count of occurrences of
    i.i.d. exponential random variables with mean :math:`1/\lambda`. This class
    generates samples of times for which cumulative exponential random
    variables occur.

    :param float rate: the parameter :math:`\lambda` which defines the rate of
        occurrences of the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        rate: float = 1,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)

        self.rate = rate

    def __str__(self) -> str:
        return f"Poisson process with rate {self.rate}."

    def __repr__(self) -> str:
        return f"PoissonProcess(rate={self.rate})"

    @property
    def rate(self) -> float:
        """Rate parameter."""
        return self.__rate

    @rate.setter
    def rate(self, value: float) -> None:
        check_nonnegative_number(value=value, name="Arrival rate")

        self.__rate = value

    def _sample_poisson_process(
        self,
        n: int | None = None,
        length: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization of a Poisson process.

        Generate a poisson process sample up to count of length if time=False,
        otherwise generate a sample up to time t=length if time=True
        """
        if n is not None:
            check_positive_integer(n=n)

            exponentials = self.rng.exponential(scale=1.0 / self.rate, size=n)

            s = np.array([0] + list(np.cumsum(exponentials)))
            return s

        if length is not None:
            check_positive_number(value=length, name="Sample length")

            t = 0
            times = [0]
            exp_rate = 1.0 / self.rate

            while t < length:
                t += self.rng.exponential(scale=exp_rate)
                times.append(t)

            return np.array(times)

        raise ValueError("Must provide either argument n or length.")

    @override
    def sample(
        self,
        n: int,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization.

        Exactly one of `n` and `length` must be provided.

        :param int n: the number of arrivals to simulate
        """
        return self._sample_poisson_process(n)
