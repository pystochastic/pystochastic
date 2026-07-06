from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ...utils.validation import check_numeric, check_positive_number
from ..noise.gaussian_noise import GaussianNoise


class BrownianMotion(GaussianNoise):
    """Brownian motion.

    .. image:: _static/brownian_motion.png
        :scale: 50%

    A standard Brownian motion (discretely sampled) has independent and
    identically distributed Gaussian increments with variance equal to
    increment length. Non-standard Brownian motion includes a linear drift
    parameter and scale factor.

    :param float drift: rate of change of the expected value
    :param float scale: scale factor of the Gaussian process
    :param float t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        drift: float = 0.0,
        scale: float = 1.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.drift = drift
        self.scale = scale

    def __str__(self) -> str:
        if self.drift == 0 and self.scale == 1:
            return f"Standard Brownian motion on interval [0, {self.t}]."

        return (
            f"Brownian motion with drift {self.drift} and "
            f"scale {self.scale} on interval [0, {self.t}]."
        )

    def __repr__(self) -> str:
        return (
            f"BrownianMotion(drift={self.drift}, "
            f"scale={self.scale}, t={self.t})"
        )

    @property
    def drift(self) -> float:
        """Drift parameter."""
        return self._drift

    @drift.setter
    def drift(self, value: float) -> None:
        check_numeric(value=value, name="Drift")

        self._drift = value

    @property
    def scale(self) -> float:
        """Scale parameter."""
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        check_positive_number(value=value, name="Scale")

        self._scale = value

    def _sample_brownian_motion(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization of Brownian Motion.

        Generate a Brownian motion realization with n increments. If zero is
        True then include W_0 = 0.
        """
        # Some opt for repeats
        if self.drift != 0:
            self.set_times_with_t(self.drift, n)

        bm = np.cumsum(self.scale * self._sample_gaussian_noise(n))
        bm = np.insert(bm, [0], 0)

        if self.drift != 0:
            return self.times + bm

        return bm

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_brownian_motion(n)

    def _sample_brownian_motion_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate a Brownian motion at specified times."""
        bm = np.cumsum(self.scale * self._sample_gaussian_noise_at(times))

        if times[0] == 0:
            bm = np.insert(bm, 0, [0])

        if self.drift != 0:
            bm += [self.drift * t for t in times]

        return bm

    @override
    def sample_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return self._sample_brownian_motion_at(times)
