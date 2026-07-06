from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from pystochastic.processes.base import BaseTimeProcess
from pystochastic.utils.validation import (
    check_positive_integer,
    check_positive_number,
    times_to_increments,
)


class GammaProcess(  # pylint: disable=too-many-instance-attributes
    BaseTimeProcess
):
    """Gamma process.

    .. image:: _static/gamma_process.png
        :scale: 50%

    A Gamma process (discretely sampled) is the summation of stationary
    independent increments which are distributed as gamma random variables.
    This class supports instantiation using the mean/variance parametrization
    or the rate/scale parametrization.

    :param float mean: mean increase per unit time; supply with
        :py:attr:`variance`
    :param float variance: variance of increase per unit time; supply with
        :py:attr:`mean`
    :param float rate: the rate of jump arrivals; supply with :py:attr:`scale`
    :param float scale: the size of the jumps; supple with :py:attr:`rate`
    :param float t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        mean: float | None = None,
        variance: float | None = None,
        rate: float | None = None,
        scale: float | None = None,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)
        if rate is None and scale is None:
            if mean is None or variance is None:
                raise ValueError(
                    "Invalid parametrization. Must provide either mean and "
                    "variance or rate and scale."
                )
            self.mean = mean
            self.variance = variance
            self.rate = mean**2.0 / variance
            self.scale = 1.0 * mean / variance
        elif mean is None and variance is None:
            if rate is None or scale is None:
                raise ValueError(
                    "Invalid parametrization. Must provide either mean and "
                    "variance or rate and scale."
                )
            self.rate = rate
            self.scale = scale
            self.mean = 1.0 * self.rate / self.scale
            self.variance = 1.0 * self.mean / self.scale
        else:
            raise ValueError(
                "Invalid parametrization. Must provide either mean and "
                "variance or rate and scale."
            )

    def __str__(self) -> str:
        return (
            f"Gamma process with rate = {self.rate} and "
            f"scale = {self.scale} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"GammaProcess(rate={self.rate}, scale={self.scale}, "
            f"t={self.t})"
        )

    @property
    def mean(self) -> float:
        """Mean increase per unit time."""
        return self.__mean

    @mean.setter
    def mean(self, value: float) -> None:
        check_positive_number(value=value, name="Mean parameter")

        self.__mean = float(value)

    @property
    def rate(self) -> float:
        """Rate of jump arrivals."""
        return self.__rate

    @rate.setter
    def rate(self, value: float) -> None:
        check_positive_number(value=value, name="Rate parameter")

        self.__rate = value

    @property
    def scale(self) -> float:
        """Scale parameter for jump sizes."""
        return self.__scale

    @scale.setter
    def scale(self, value: float) -> None:
        check_positive_number(value=value, name="Scale parameter")

        self.__scale = value

    @property
    def variance(self) -> float:
        """Variance of increase per unit time."""
        return self.__variance

    @variance.setter
    def variance(self, value: float) -> None:
        check_positive_number(value=value, name="Variance parameter")

        self.__variance = value

    def _sample_gamma_process(self, n: int) -> npt.NDArray[np.float64]:
        """Sample a Gamma process."""
        check_positive_integer(n=n, name="Number of increments")

        delta_t = 1.0 * self.t / n

        shape = 1.0 * self.mean**2 * delta_t / self.variance
        scale = 1.0 * self.variance / self.mean

        samples = np.cumsum(self.rng.gamma(shape=shape, scale=scale, size=n))
        return np.concatenate(([0], samples))

    def _sample_gamma_process_at(
        self, times: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Sample a Gamma process at specific times."""
        s = []
        if times[0] != 0:
            times = np.insert(times, 0, [0])
        else:
            s.append(0)
        increments = times_to_increments(times=times)

        scale = self.variance / self.mean
        shape_coef = self.mean**2 / self.variance

        for inc in increments:
            s.append(self.rng.gamma(shape=shape_coef * inc, scale=scale))

        return np.cumsum(s)

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_gamma_process(n)

    @override
    def sample_at(
        self, times: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return self._sample_gamma_process_at(times)
