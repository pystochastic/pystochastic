import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ...utils.validation import check_positive_integer
from ..base import BaseSequenceProcess


class BernoulliProcess(BaseSequenceProcess):
    """Bernoulli process.

    .. image:: _static/bernoulli_process.png
        :scale: 50%

    A Bernoulli process consists of a sequence of Bernoulli random
    variables. A Bernoulli random variable is

    * 1 with probability :math:`p`
    * 0 with probability :math:`1-p`

    :param p: in :math:`[0,1]`, the probability of success of each Bernoulli
        random variable
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(
        self,
        p: float = 0.5,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)
        self.p = p

    def __str__(self) -> str:
        return f"Bernoulli process with p={self.p}."

    def __repr__(self) -> str:
        return f"BernoulliProcess({self.p})"

    @property
    def p(self) -> float:
        """Probability of success."""
        return self.__p

    @p.setter
    def p(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("Probability of success must be a number between 0 and 1.")
        if value < 0 or value > 1:
            raise ValueError("Probability of success p must be between 0 and 1.")
        self.__p = float(value)

    def _sample_bernoulli(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a Bernoulli process realization."""
        check_positive_integer(n)

        return np.array(
            [1.0 if trial < self.p else 0.0 for trial in self.rng.uniform(size=n)]
        )

    def sample(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a Bernoulli process realization.

        :param int n: the number of steps to simulate.
        """
        return self._sample_bernoulli(n)
