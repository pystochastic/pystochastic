import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ....utils.validation import check_numeric, check_positive_integer
from ...base import BaseTimeProcess


class ColoredNoise(BaseTimeProcess):
    r"""Colored noise processes.

    .. image:: _static/colored_noise.png
        :scale: 50%

    Also referred to as power law noise, colored noise refers to noise
    processes with power law spectral density. That is, their spectral density
    per unit bandwidth is proportional to :math:`(1/f)^\beta`, where
    :math:`f` is frequency with exponent :math:`\beta`.

    Uses the algorithm from:

    * Timmer, J., and M. Koenig. "On generating power law noise."
      Astronomy and Astrophysics 300 (1995): 707.

    Generates a normalized power-law spectral noise.

    :param float beta: the power law exponent for the spectral density, with 0
        being white noise, 1 being pink noise, 2 being red noise (Brownian
        noise), -1 being blue noise, -2 being violet noise. Default is 0
        (white noise).
    :param float t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(
        self,
        beta: float = 0,
        t: float = 1,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)
        self.beta = beta
        self.__n: int | None = None
        self.__half: int | None = None
        self.__frequencies: npt.NDArray[np.float64] | None = None
        self.__scale: float | None = None

    def __str__(self) -> str:
        return (
            "Colored noise generator with exponent "
            f"{self.beta} on interval [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return f"ColoredNoise(beta={self.beta}, t={self.t})"

    @property
    def beta(self) -> float:
        """Power law exponent."""
        return self.__beta

    @beta.setter
    def beta(self, value: float) -> None:
        check_numeric(value, "beta")
        self.__beta = value

    def _sample_colored_noise(self, n: int) -> npt.NDArray[np.float64]:
        """Generate colored noise increments at specified times from zero."""
        check_positive_integer(n)
        n = n + 1
        if self.__n != n:
            self.__n = n

            self.__half = (n + 1) // 2
            self.__frequencies = np.fft.fftfreq(n, self.t)
            self.__scale = [
                np.sqrt(0.5 * (1 / w) ** self.beta)
                for w in self.__frequencies[1 : self.__half]
            ]

        assert self.__half is not None

        gn_real = np.random.normal(size=self.__half - 1)
        gn_imag = np.random.normal(size=self.__half - 1)
        fft = self.__scale * (gn_real + 1j * gn_imag)

        if n % 2 == 0:
            f = np.concatenate(
                (
                    [0],
                    fft,
                    [
                        np.sqrt(
                            0.5 * (1 / -self.__frequencies[self.__half]) ** self.beta
                        )
                        * np.random.normal()
                    ],
                    np.conj(fft)[::-1],
                )
            )
        else:
            f = np.concatenate(([0], fft, np.conj(fft)[::-1]))

        return np.fft.ifft(f).real / np.std(f)

    def sample(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization of colored noise.

        Generate a colored noise realization with n increments.

        :param int n: the number of increments to generate.
        """
        return self._sample_colored_noise(n)
