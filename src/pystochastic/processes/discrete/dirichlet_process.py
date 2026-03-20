from typing import Callable, override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ...utils.validation import check_positive_number
from ..base import BaseSequenceProcess


class DirichletProcess(BaseSequenceProcess):
    r"""Dirichlet process.

    .. image:: _static/dirichlet_process.png
        :scale: 50%

    A Dirichlet process is a stochastic process in which the resulting samples
    can be interpreted as discrete probability distributions.

    For each step :math:`k \geq 1`, draw from the base distribution with
    probability

    .. math::
        \frac{\alpha}{\alpha + k - 1}

    Otherwise draw randomly from the previous steps.

    :param callable base: a zero argument callable used as the base
        distribution sampler. The default base distribution is Uniform(0, 1).
    :param float alpha: a non-negative value used to determine probability of
        drawing a new value from the base distribution
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(
        self,
        base: Callable[[], float] | None = None,
        alpha: float = 1,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)

        if base is None:
            base = self.rng.uniform

        self.base = base
        self.alpha = alpha

    @property
    def base(self) -> Callable[[], float]:
        """The base distribution callable for sampling new step values."""
        return self.__base

    @base.setter
    def base(self, value: Callable[[], float]) -> None:
        if not callable(value):
            raise ValueError("base must be callable")
        self.__base = value

    @property
    def alpha(self) -> float:
        """Parameter for determining the probability of sampling new values."""
        return self.__alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        check_positive_number(value)
        self.__alpha = value

    def _sample(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a realization of the Dirichlet process.

        :param int n: the number of steps of the Dirichlet process to generate.
        """
        sequence = []
        for k in range(1, n + 1):
            draw_proba = self.alpha / (self.alpha + k - 1)
            if self.rng.uniform() < draw_proba:
                sequence.append(self.base())
            else:
                sequence.append(self.rng.choice(sequence))
        return np.array(sequence)

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample(n)
