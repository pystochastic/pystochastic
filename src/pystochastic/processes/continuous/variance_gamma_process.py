from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ...utils.validation import (
    check_numeric,
    check_positive_integer,
    check_positive_number,
)
from ..noise import GaussianNoise


class VarianceGammaProcess(GaussianNoise):
    r"""Variance Gamma process.

    .. image:: _static/variance_gamma_process.png
        :scale: 50%

    A variance gamma process has independent increments which follow the
    variance-gamma distribution. It can be represented as a Brownian motion
    with drift subordinated by a Gamma process:

    .. math::

        \theta \Gamma(t; 1, \nu) + \sigma W(\Gamma(t; 1, \nu))

    :param float drift: the drift parameter of the Brownian motion,
        or :math:`\theta` above
    :param float variance: the variance parameter of the Gamma subordinator,
        or :math:`\nu` above
    :param float scale: the scale parameter of the Brownian motion,
        or :math:`\sigma` above
    :param float t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        drift: float = 0.0,
        variance: float = 1.0,
        scale: float = 1.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.drift = drift
        self.variance = variance
        self.scale = scale

    def __str__(self) -> str:
        return (
            f"Variance Gamma process on interval [0, {self.t}] with drift "
            f"{self.drift}, variance {self.variance}, and scale {self.scale}."
        )

    def __repr__(self) -> str:
        return (
            f"VarianceGammaProcess(drift={self.drift}, variance={self.variance}, "
            f"scale={self.scale}, t={self.t})"
        )

    @property
    def drift(self) -> float:
        """Drift parameter."""
        return self.__drift

    @drift.setter
    def drift(self, value: float) -> None:
        check_numeric(value=value, name="Drift")

        self.__drift = value

    @property
    def variance(self) -> float:
        """Variance parameter."""
        return self.__variance

    @variance.setter
    def variance(self, value: float) -> None:
        check_positive_number(value=value, name="Variance")
        self.__variance = value

    @property
    def scale(self) -> float:
        """Scale parameter."""
        return self.__scale

    @scale.setter
    def scale(self, value: float) -> None:
        check_positive_number(value=value, name="Scale")
        self.__scale = value

    def _sample_variance_gamma_process(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization of a variance gamma process."""
        check_positive_integer(n=n, name="n")

        delta_t = 1.0 * self.t / n
        shape = delta_t / self.variance
        scale = self.variance

        gammas = self.rng.gamma(shape=shape, scale=scale, size=n)
        gn = super().sample(n)

        increments = self.drift * gammas + self.scale * np.sqrt(gammas) * gn

        samples = np.cumsum(increments)

        return np.concatenate(([0], samples))

    def _sample_variance_gamma_process_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate a realization of a variance gamma process."""
        if times[0] != 0:
            zero = False
            times = np.array([0] + list(times))
        else:
            zero = True

        shapes = np.diff(times) / self.variance
        scale = self.variance

        gammas = np.array(
            [
                self.rng.gamma(
                    shape=shape,
                    scale=scale,
                    size=1,
                )[0]
                for shape in shapes
            ]
        )
        gn = super().sample_at(times)

        increments = self.drift * gammas + self.scale * np.sqrt(gammas) * gn

        samples = np.cumsum(increments)
        if zero:
            samples = np.insert(samples, 0, [0])
        return samples

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_variance_gamma_process(n)

    @override
    def sample_at(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self._sample_variance_gamma_process_at(times)
