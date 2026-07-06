from typing import override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from ...utils.validation import check_numeric, check_positive_integer
from ..base import BaseSequenceProcess


class ChineseRestaurantProcess(BaseSequenceProcess):
    """Chinese restaurant process.

    .. image:: _static/chinese_restaurant_process.png
        :scale: 50%

    A Chinese restaurant process consists of a sequence of arrivals of
    customers to a Chinese restaurant. Customers may be seated either at an
    occupied table or a new table, there being infinitely many customers
    and tables.

    The first customer sits at the first table. The :math:`n`-th customer
    sits at a new table with probability :math:`1/n`, and at each already
    occupied table with probability :math:`t_k/n`, where :math:`t_k` is the
    number of customers already seated at table :math:`k`. This is the
    canonical process with :math:`discount=0` and :math:`strength=1`.

    The generalized process gives the :math:`n`-th customer a probability of
    :math:`(strength + T * discount) / (n - 1 + strength)` to sit at a new
    table and a probability of :math:`(t_k - discount) / (n - 1 + strength)`
    of sitting at table :math:`k`. :math:`T` is the number of occupied tables.

    Samples provide a sequence of tables selected by a sequence of customers.

    :param float discount: the discount value of existing tables.
        Must be strictly less than 1.
    :param float strength: the strength of a new table. If discount
        is negative, strength must be a multiple of discount. If discount is
        nonnegative, strength must be strictly greater than the
        negative discount.
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(
        self,
        discount: float = 0.0,
        strength: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)

        self.discount = discount
        self.strength = strength

    def __str__(self) -> str:
        return (
            f"Chinese restaurant process with discount={self.discount} and strength={self.strength}"
        )

    def __repr__(self) -> str:
        return f"ChineseRestaurantProcess(discount={self.discount}, strength={self.strength})"

    @property
    def discount(self) -> float:
        """Discount parameter."""
        return self.__discount

    @discount.setter
    def discount(self, value: float | int) -> None:
        check_numeric(value=value, name="Discount")

        if value >= 1.0:
            raise ValueError(f"Discount value must be less than 1; got {value}.")

        self.__discount = value

    @property
    def strength(self) -> float:
        """Strength parameter."""
        return self.__strength

    @strength.setter
    def strength(self, value: float | int) -> None:
        check_numeric(value=value, name="Strength")

        if self.discount < 0:
            strength_positive = 1.0 * value / -self.discount <= 0
            strength_not_multiple = (1.0 * value / -self.discount) % 1 != 0
            if strength_positive or strength_not_multiple:
                raise ValueError(
                    "When discount is negative, strength value must be equal to a multiple of"
                    f" the discount value; got {value}."
                )
        elif self.discount < 1:
            if value <= -self.discount:
                raise ValueError(
                    "When discount is between 0 and 1, strength value must be greater than the "
                    f"negative of the discount; got {value}."
                )
        self.__strength = value

    def _sample_chinese_restaurant(
        self,
        n: int,
        partition: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Generate a Chinese restaurant process with n customers."""
        check_positive_integer(n=n)

        c: list[list[float]] = [[1]]
        s = [0]
        num_tables = 1
        table_range = [0, 1]

        for k in range(2, n + 1):
            p = [
                1.0 * (len(c[t]) - self.discount) / (k - 1 + self.strength)
                for t in table_range[:-1]
            ]
            p.append(1.0 * (self.strength + num_tables * self.discount) / (k - 1 + self.strength))
            table = self.rng.choice(table_range, p=p)
            if table == num_tables:
                num_tables += 1
                table_range.append(num_tables)
                c.append([])
            c[table].append(k - 1)
            s.append(table)

        if partition:
            return np.array([np.array(t) for t in c], dtype=object)

        return np.array(s)

    @override
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        return self._sample_chinese_restaurant(n)

    def sample_partition(self, n: int) -> npt.NDArray[np.float64]:
        """Generate a Chinese restaurant process partition.

        :param n: the number of customers to simulate.
        """
        return self._sample_chinese_restaurant(n, partition=True)

    def sequence_to_partition(
        self,
        sequence: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Create a partition from a sequence.

        :param sequence: a Chinese restaurant sample.
        """
        partition: list[list[float]] = []
        tables = -1
        for idx, table in enumerate(sequence):
            if table > tables:
                tables = table
                partition.append([])
            partition[table].append(idx)

        return np.array([np.array(t) for t in partition], dtype=object)

    def partition_to_sequence(
        self,
        partition: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Create a sequence from a partition.

        :param partition: a Chinese restaurant partition.
        """
        length = 0
        for table in partition:
            length += len(table)
        sequence = [0] * length
        for idx, table in enumerate(partition):
            for c in table:
                sequence[c] = idx

        return np.array(sequence)
