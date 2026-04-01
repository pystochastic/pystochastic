from typing import Iterable, override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ...utils.validation import check_positive_integer
from ..base import BaseSequenceProcess

DEFAULT_TRANSITION: npt.NDArray[np.float64] = np.array(
    [[0.5, 0.5], [0.5, 0.5]],
    dtype=np.float64,
)


class MarkovChain(BaseSequenceProcess):
    """Finite state Markov chain.

    .. image:: _static/markov_chain.png
        :scale: 50%

    A Markov Chain which changes between states according to the transition
    matrix.

    :param 2darray transition: a square matrix representing the transition
        probabilities between states.
    :param 1darray initial: a vector representing the initial state probabilities. If
        not provided, each state has equal initial probability.
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(
        self,
        transition: Iterable[Iterable[float]] | npt.NDArray[np.float64] = DEFAULT_TRANSITION,
        initial: Iterable[float] | npt.NDArray[np.float64] | None = None,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)

        self.transition = transition

        if initial is None:
            initial = [float(1.0 / len(self.transition)) for _ in self.transition]

        self.initial = initial

        self.__num_states = len(self.__initial)

    def __str__(self) -> str:
        return (
            f"Markov chain with transition matrix = {self.transition} "
            f"and initial state probabilities = {self.initial}"
        )

    def __repr__(self) -> str:
        return f"MarkovChain(transition={self.transition}, initial={self.initial})"

    @property
    def transition(self) -> npt.NDArray[np.float64]:
        """Transition probability matrix."""
        return self.__transition

    @transition.setter
    def transition(self, values: Iterable[Iterable[float]] | npt.NDArray[np.float64]) -> None:
        values = np.asarray(values, dtype=np.float64)

        if values.ndim != 2:
            raise ValueError(
                "Transition matrix must be a two-dimensional array; "
                f"got {values.ndim} dimensions."
            )

        if values.shape[0] != values.shape[1]:
            raise ValueError(
                "Transition matrix must be a square matrix; "
                f"got {values.shape[0]}x{values.shape[1]}."
            )

        for row in values:
            if not np.isclose(np.sum(row), 1.0):
                raise ValueError(
                    "Transition matrix is not a proper stochastic matrix; "
                    f"row sums to {np.sum(row)}."
                )

        self.__transition = values

    @property
    def initial(self) -> npt.NDArray[np.float64]:
        """Vector of initial state probabilities."""
        return self.__initial

    @initial.setter
    def initial(self, values: Iterable[float] | npt.NDArray[np.float64]) -> None:
        values = np.asarray(values, dtype=np.float64)

        if values.ndim != 1:
            raise ValueError(
                "Initial state probabilities must be a one-dimensional vector; "
                f"got {values.ndim} dimensions."
            )

        if len(values) != len(self.transition):
            raise ValueError(
                "Length of initial state probabilities must match the number of states in the "
                f"transition matrix; got {len(values)} values for {len(self.transition)} states."
            )

        if not np.isclose(np.sum(values), 1.0):
            raise ValueError(f"Initial state probabilities must sum to 1; got {np.sum(values)}.")

        self.__initial = values

    @property
    def num_states(self) -> int:
        """Number of states in the Markov chain."""
        return self.__num_states

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        check_positive_integer(n=n)

        states = range(self.num_states)

        markov_chain = [self.rng.choice(states, p=self.initial)]
        for _ in range(n - 1):
            markov_chain.append(self.rng.choice(states, p=self.transition[markov_chain[-1]]))

        return np.array(markov_chain)
