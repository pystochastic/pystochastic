from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from scipy.stats import levy

from ...utils.validation import check_positive_integer
from .brownian_motion import BrownianMotion


class CauchyProcess(BrownianMotion):
    """Symmetric Cauchy process.

    .. image:: _static/cauchy_process.png
        :scale: 50%

    The symmetric Cauchy process is a Brownian motion with a Levy subordinator
    using location parameter 0 and scale parameter :math:`t^2/2`.

    :param float t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(
            drift=0.0,
            scale=1.0,
            t=t,
            rng=rng,
        )

    def __str__(self) -> str:
        return f"Cauchy process on [0, {self.t}]"

    def __repr__(self) -> str:
        return f"CauchyProcess(t={self.t})"

    def _sample_cauchy_process(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization of a Cauchy process."""
        check_positive_integer(n=n, name="Number of increments")

        delta_t = 1.0 * self.t / n

        times = np.cumsum(
            levy.rvs(
                loc=0,
                scale=delta_t**2 / 2,
                size=n,
                random_state=self.rng,
            )
        )

        times = np.insert(times, 0, [0])
        return self._sample_brownian_motion_at(times)

    def _sample_cauchy_process_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate a realization of a Cauchy process."""
        if times[0] != 0:
            zero = False
            times = np.insert(times, 0, [0])
        else:
            zero = True

        deltas = np.diff(times)
        levys = [
            levy.rvs(
                loc=0,
                scale=d**2 / 2,
                size=1,
                random_state=self.rng,
            )
            for d in deltas
        ]

        ts: npt.NDArray[np.float64] = np.cumsum(
            levys,  # type: ignore[arg-type]
        )

        if zero:
            ts = np.insert(ts, 0, [0])

        return self._sample_brownian_motion_at(ts)

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_cauchy_process(n)

    @override
    def sample_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return self._sample_cauchy_process_at(times)
