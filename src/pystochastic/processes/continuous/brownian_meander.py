from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ...utils.validation import check_nonnegative_number
from .brownian_bridge import BrownianBridge


class BrownianMeander(BrownianBridge):
    """Brownian meander process.

    .. image:: _static/brownian_meander.png
        :scale: 50%

    A Brownian motion conditioned such that the process is nonnegative.

    Generated using method by

    * Williams, David. "Decomposing the Brownian path." Bulletin of the
      American Mathematical Society 76, no. 4 (1970): 871-873.

    * Imhof, J-P. "Density factorizations for Brownian motion, meander and the
      three-dimensional Bessel process, and applications." Journal of Applied
      Probability 21, no. 3 (1984): 500-510.

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
            b=0.0,
            t=t,
            rng=rng,
        )

    def __str__(self) -> str:
        return f"Brownian meander on [0, {self.t}]"

    def __repr__(self) -> str:
        return f"BrownianMeander(t={self.t})"

    def _sample_brownian_meander(
        self,
        n: int,
        b: float | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""Generate a Brownian meander realization.

        Williams, 1970, or Imhof, 1984.

        :param int n: the number of increments to generate
        :param float b: the nonnegative right hand endpoint of the meander. If
            not provided, one is randomly selected from a :math:`\sqrt{2E}`
            random variable where :math:`E` is exponential.
        """
        if b is None:
            b = np.sqrt(2 * self.t * self.rng.exponential())
        else:
            check_nonnegative_number(value=b, name="Right endpoint")

        self.set_times(n)

        bridge_1 = self._sample_brownian_bridge(n)
        bridge_2 = self._sample_brownian_bridge(n)
        bridge_3 = self._sample_brownian_bridge(n)

        ret = np.sqrt(
            (b * self.times / self.t + bridge_1) ** 2
            + bridge_2**2
            + bridge_3**2
        )

        return ret

    def _sample_brownian_meander_at(
        self,
        times: npt.NDArray[np.float64],
        b: float | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""Generate a Brownian meander realization.

        Williams, 1970, or Imhof, 1984.
        :param times: a vector of increasing time values at which to generate
            the realization
        :param float b: the right endpoint value for :py:attr:`times` [-1]. If
            not provided, one is randomly selected from a :math:`\sqrt{2tE}`
            random variable where :math:`E` is exponential and :math:`t` is
            :py:attr:`times` [-1].
        """
        if b is None:
            b = np.sqrt(2 * times[-1] * self.rng.exponential())
        else:
            check_nonnegative_number(value=b, name="Right endpoint")

        bridge_1 = self._sample_brownian_bridge_at(times)
        bridge_2 = self._sample_brownian_bridge_at(times)
        bridge_3 = self._sample_brownian_bridge_at(times)

        ret = np.sqrt(
            (b * times / times[-1] + bridge_1) ** 2 + bridge_2**2 + bridge_3**2
        )

        assert isinstance(ret, np.ndarray)

        return ret

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_brownian_meander(n)

    @override
    def sample_at(
        self,
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return self._sample_brownian_meander_at(times)

    def sample_with_endpoint(
        self,
        n: int,
        b: float,
    ) -> npt.NDArray[np.float64]:
        """Generate a Brownian meander realization with specified endpoint.

        :param int n: the number of increments to generate
        :param float b: the nonnegative right hand endpoint of the meander.
        """
        return self._sample_brownian_meander(n, b)

    def sample_at_with_endpoint(
        self,
        times: npt.NDArray[np.float64],
        b: float,
    ) -> npt.NDArray[np.float64]:
        """Generate a Brownian meander realization with specified endpoint.

        :param times: a vector of increasing time values at which to generate
            the realization
        :param float b: the nonnegative right endpoint of the meander.
        """
        return self._sample_brownian_meander_at(times, b)
