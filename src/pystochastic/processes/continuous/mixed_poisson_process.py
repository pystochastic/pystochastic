from typing import Any, Callable, override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .poisson_process import PoissonProcess


class MixedPoissonProcess(PoissonProcess):
    r"""Mixed poisson process.

    .. image:: _static/mixed_poisson_process.png
        :scale: 50%

    A mixed poisson process is a Poisson process for which the rate is
    a scalar random variate. The sample method will generate a random variate
    for the rate before generating a Poisson process realization with the rate.
    A Poisson process with rate :math:`\lambda`
    is a count of occurrences of i.i.d. exponential random
    variables with mean :math:`1/\lambda`. Use the ``rate`` attribute to get
    the most recently generated random rate.

    :param callable rate_func: a callable to generate variates of the random
        rate
    :param tuple rate_args: positional args for ``rate_func``
    :param dict rate_kwargs: keyword args for ``rate_func``
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        rate_func: Callable[..., float],
        rate_args: tuple[float, ...] | None = None,
        rate_kwargs: dict[str, Any] | None = None,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(
            rate=1.0,
            rng=rng,
        )

        self.rate_func = rate_func
        self.rate_args = rate_args or tuple()
        self.rate_kwargs = rate_kwargs or {}

    def __str__(self) -> str:
        return "Mixed Poisson process with random rate."

    def __repr__(self) -> str:
        return (
            f"MixedPoissonProcess(rate_func={self.rate_func}, "
            f"rate_args={self.rate_args}, rate_kwargs={self.rate_kwargs})"
        )

    @property
    def rate_func(self) -> Callable[..., float]:
        """Current rate's distribution."""
        return self.__rate_func

    @rate_func.setter
    def rate_func(self, value: Callable[..., float]) -> None:
        if not callable(value):
            raise TypeError("Rate function must be a callable.")
        self.__rate_func = value

    @property
    def rate_args(self) -> tuple[float, ...]:
        """Positional arguments for the rate function."""
        return self.__rate_args

    @rate_args.setter
    def rate_args(self, value: tuple[float, ...]) -> None:
        if not isinstance(value, (list, tuple)):
            raise TypeError("Rate args must be a list or tuple.")
        self.__rate_args = tuple(value)

    @property
    def rate_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the rate function."""
        return self.__rate_kwargs

    @rate_kwargs.setter
    def rate_kwargs(self, value: dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise TypeError("Rate kwargs must be a dict.")
        self.__rate_kwargs = value

    def _sample_rate(self) -> float:
        """Generate a rate variate."""
        return self.rate_func(
            *self.rate_args,
            **self.rate_kwargs,
        )

    @override
    def sample(
        self,
        n: int,
    ) -> npt.NDArray[np.float64]:
        self.rate = self._sample_rate()
        return self._sample_poisson_process(n=n)

    @override
    def sample_with_length(
        self,
        length: float,
    ) -> npt.NDArray[np.float64]:
        self.rate = self._sample_rate()
        return self._sample_poisson_process(length=length)
