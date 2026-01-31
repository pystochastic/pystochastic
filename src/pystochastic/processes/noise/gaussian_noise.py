from typing import override

import numpy as np
import numpy.typing as npt

from ...utils.validation import check_positive_integer, times_to_increments
from ..base import BaseTimeProcess


class GaussianNoise(BaseTimeProcess):
    """Gaussian noise process.

    .. image:: _static/gaussian_noise.png
        :scale: 50%

    Generate a sequence of Gaussian random variables.

    :param float t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __str__(self) -> str:
        return f"Gaussian noise generator on interval [0, {self.t}]"

    def __repr__(self) -> str:
        return f"GaussianNoise(t={self.t})"

    def _sample_gaussian_noise(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization of Gaussian noise.

        Generate a Gaussian noise realization with n increments.
        """
        check_positive_integer(n=n, name="Number of increments")

        delta_t = 1.0 * self.t / n

        noise = self.rng.normal(
            scale=np.sqrt(delta_t),
            size=n,
        )

        return noise

    def _sample_gaussian_noise_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate Gaussian noise increments at specified times from zero."""
        if times[0] != 0:
            times = np.concatenate(([0], times))

        increments = times_to_increments(times=times)

        noise = np.array(
            [
                self.rng.normal(
                    scale=np.sqrt(inc),
                )
                for inc in increments
            ]
        )

        return noise

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_gaussian_noise(n)

    @override
    def sample_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return self._sample_gaussian_noise_at(times)
