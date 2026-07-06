from typing import override

import numpy as np
import numpy.typing as npt

from ..noise import FractionalGaussianNoise


class FractionalBrownianMotion(FractionalGaussianNoise):
    """Fractional Brownian motion process.

    .. image:: _static/fractional_brownian_motion.png
        :scale: 50%

    A fractional Brownian motion (discretely sampled) has correlated Gaussian
    increments defined by Hurst parameter :math:`H`. When :math:`H = 1/2`,
    the process is a standard Brownian motion. When :math:`H > 1/2`, the
    increments are positively correlated. When :math:`H < 1/2`, the
    increments are negatively correlated.

    Hosking's method:

    * Hosking, Jonathan RM. "Modeling persistence in hydrological time series
      using fractional differencing." Water resources research 20, no. 12
      (1984): 1898-1908.

    Davies Harte method:

    * Davies, Robert B., and D. S. Harte. "Tests for Hurst effect." Biometrika
      74, no. 1 (1987): 95-101.

    :param float hurst: the Hurst parameter on the interval (0, 1)
    :param float t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __str__(self) -> str:
        return (
            f"Fractional Brownian motion with Hurst {self.hurst} on "
            f"[0, {self.t}]."
        )

    def __repr__(self) -> str:
        return f"FractionalBrownianMotion(hurst={self.hurst}, t={self.t})"

    def _sample_fractional_brownian_motion(
        self,
        n: int,
    ) -> npt.NDArray[np.float64]:
        """Generate a realization of fractional Brownian motion."""
        fgn = self.sample_davies_harte(n)
        fbm = fgn.cumsum()
        fbm = np.insert(fbm, [0], 0)
        return fbm

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization.

        :param int n: the number of increments to generate
        """
        return self._sample_fractional_brownian_motion(n)
