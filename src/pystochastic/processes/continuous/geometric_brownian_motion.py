from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ...utils.validation import (
    check_numeric,
    check_positive_integer,
    check_positive_number,
)
from .brownian_motion import BrownianMotion


class GeometricBrownianMotion(BrownianMotion):
    r"""Geometric Brownian motion process.

    .. image:: _static/geometric_brownian_motion.png
        :scale: 50%

    A geometric Brownian motion :math:`S_t` is the analytic solution to the
    stochastic differential equation with Wiener process :math:`W_t`:

    .. math::

        dS_t = \mu S_t dt + \sigma S_t dW_t

    and can be represented with initial value :math:`S_0` in the form:

    .. math::

        S_t = S_0 \exp \left( \left( \mu - \frac{\sigma^2}{2} \right) t +
        \sigma W_t \right)

    :param float drift: the parameter :math:`\mu`
    :param float volatility: the parameter :math:`\sigma`
    :param float t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        drift: float = 0,
        volatility: float = 1,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.drift = drift
        self.volatility = volatility

    def __str__(self) -> str:
        return (
            f"Geometric Brownian motion with drift {self.drift} "
            f"and volatility {self.volatility} on [0, {self.t}]."
        )

    def __repr__(self) -> str:
        return (
            f"GeometricBrownianMotion(drift={self.drift}, "
            f"volatility={self.volatility}, t={self.t})"
        )

    @property
    def drift(self) -> float:
        """Geometric Brownian motion drift parameter."""
        return self.__drift

    @drift.setter
    def drift(self, value: float) -> None:
        check_numeric(value=value, name="Drift")

        self.__drift = value

    @property
    def volatility(self) -> float:
        """Geometric Brownian motion volatility parameter."""
        return self.__volatility

    @volatility.setter
    def volatility(self, value: float) -> None:
        check_positive_number(value=value, name="Volatility")

        self.__volatility = value

    def _sample_geometric_brownian_motion(
        self,
        n: int,
        initial: float = 1.0,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization of geometric Brownian motion."""
        check_positive_integer(n=n)
        check_positive_number(value=initial, name="Initial")

        # Opt for repeated use
        self.set_times_with_t(self.drift - self.volatility**2 / 2.0, n)

        noise = self.volatility * self._sample_brownian_motion(n)

        return initial * np.exp(self.times + noise)

    def _sample_geometric_brownian_motion_at(
        self,
        times: npt.NDArray[np.float64],
        initial: float = 1.0,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization of geometric Brownian motion."""
        line = [(self.drift - self.volatility**2 / 2.0) * t for t in times]
        noise = self.volatility * self._sample_brownian_motion_at(times)

        ret = initial * np.exp(line + noise)
        assert isinstance(ret, np.ndarray)

        return ret

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_geometric_brownian_motion(n, 1.0)

    @override
    def sample_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return self._sample_geometric_brownian_motion_at(times, 1.0)

    def sample_with_initial(
        self,
        n: int,
        initial: float = 1.0,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization.

        :param int n: the number of increments to generate.
        :param float initial: the initial value of the process :math:`S_0`.
        """
        return self._sample_geometric_brownian_motion(n, initial)

    def sample_at_with_initial(
        self,
        times: npt.NDArray[np.float64],
        initial: float = 1.0,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization using specified times.

        :param times: a vector of increasing time values at which to generate
            the realization
        :param float initial: the initial value of the process :math:`S_0`.
        """
        return self._sample_geometric_brownian_motion_at(times, initial)
