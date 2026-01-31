from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ...utils.validation import (
    check_nonnegative_integer,
    check_positive_integer,
)
from .brownian_motion import BrownianMotion


class BesselProcess(BrownianMotion):
    r"""Bessel process.

    .. image:: _static/bessel_process.png
        :scale: 50%

    The Bessel process is the Euclidean norm of an :math:`n`-dimensional
    Wiener process, e.g. :math:`\|\mathbf{W}_t\|`

    Generate Bessel process realizations using :py:attr:`dim` independent
    Brownian motion processes on the interval :math:`[0,t]`

    :param int dim: the number of underlying independent Brownian motions to
        use
    :param float t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        dim: int = 1,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(t=t, rng=rng)

        self.dim = dim

    def __str__(self) -> str:
        return (
            f"Bessel process of {self.dim} Wiener processes on "
            f"[0, {self.t}]"
        )

    def __repr__(self) -> str:
        return f"BesselProcess(dim={self.dim}, t={self.t})"

    @property
    def dim(self) -> int:
        """Dimensions, or independent Brownian motions."""
        return self._dim

    @dim.setter
    def dim(self, value: int) -> None:
        check_nonnegative_integer(n=value, name="Dimension")

        self._dim = value

    def _sample_bessel_process(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization of a Bessel process."""
        check_positive_integer(n=n, name="Number of increments")

        samples = [self._sample_brownian_motion(n) for _ in range(self.dim)]

        return np.array([np.linalg.norm(coord) for coord in zip(*samples)])

    def _sample_bessel_process_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate a realization of a Bessel process."""
        samples = [
            self._sample_brownian_motion_at(times) for _ in range(self.dim)
        ]
        return np.array([np.linalg.norm(coord) for coord in zip(*samples)])

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_bessel_process(n)

    @override
    def sample_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return self._sample_bessel_process_at(times)
