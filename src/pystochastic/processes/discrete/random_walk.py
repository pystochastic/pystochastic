from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ...utils.validation import (
    check_nonnegative_number,
    check_numeric,
    check_positive_integer,
)
from ..base import BaseSequenceProcess


class RandomWalk(BaseSequenceProcess):
    """Random walk.

    .. image:: _static/random_walk.png
        :scale: 50%

    A random walk is a sequence of random steps taken from a set of step sizes
    with a probability distribution. By default this object defines the steps
    to be [-1, 1] with probability 1/2 for each possibility.

    :param steps: a vector of possible deltas to apply at each step.
    :param weights: a corresponding vector of weights associated with each
        step value. If not provided each step has equal weight/probability.
    """

    def __init__(
        self,
        *,
        steps: list[float | int] | npt.NDArray[np.float64] | None = None,
        weights: list[float | int] | npt.NDArray[np.float64] | None = None,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)
        self.steps = steps or [-1.0, 1.0]

        length = len(self.steps)
        if length < 1:
            raise ValueError("Steps must have at least one element.")

        if weights is None:
            self.weights = [1.0 for _ in self.steps]
            self.p = [1.0 / length for _ in self.steps]
        else:
            if len(weights) != length:
                raise ValueError("Steps and probabilities must have same length.")

            self.weights = weights
            total = sum(weights)
            self.p = [1.0 * w / total for w in weights]

    def __str__(self) -> str:
        return f"Random walk steps = {self.steps} and weights = {self.weights}"

    def __repr__(self) -> str:
        return f"RandomWalk(steps={self.steps}, weights={self.weights})"

    @property
    def p(self) -> npt.NDArray[np.float64]:
        """Step probabilities, normalized from :py:attr:`weights`."""
        return self.__p

    @p.setter
    def p(self, values: list[float] | npt.NDArray[np.float64]) -> None:
        self.__p = np.array(values, dtype=np.float64, copy=True)

    @property
    def steps(self) -> npt.NDArray[np.float64]:
        """Possible steps."""
        return self.__steps

    @steps.setter
    def steps(self, values: list[float | int] | npt.NDArray[np.float64]) -> None:
        for i, value in enumerate(values):
            check_numeric(value=value, name=f"Step values[{i}] = {value}")

        self.__steps = np.array(values, dtype=np.float64, copy=True)

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Step weights provided."""
        return self.__weights

    @weights.setter
    def weights(self, values: list[float | int] | npt.NDArray[np.float64]) -> None:
        for i, value in enumerate(values):
            check_nonnegative_number(value=value, name=f"Weight values[{i}] = {value}")

        self.__weights = np.array(values, dtype=np.float64, copy=True)

    def _sample_random_walk(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a random walk."""
        return np.array([0] + list(np.cumsum(self._sample_random_walk_increments(n))))

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_random_walk(n)

    def _sample_random_walk_increments(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a sample of random walk increments."""
        check_positive_integer(n=n)
        return self.rng.choice(self.steps, p=self.p, size=n)

    def sample_increments(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a sample of random walk increments.

        :param int n: the number of increments to generate.
        """
        return self._sample_random_walk_increments(n)
