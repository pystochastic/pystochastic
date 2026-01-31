import numpy as np

_default_rng = np.random.default_rng()

#: The default random number generator
generator = _default_rng


def use_generator(
    *,
    rng: np.random.Generator | None = None,
) -> None:
    """Use the new numpy Generator as default for pystochastic.

    Sets the default random number generator for stochastic processes to
    the newer ``np.random.default_rng()``.

    .. note::

        This is the default generator and
        there is no need to call this function unless returning to the default
        after switching away from it.

    :param numpy.random.Generator rng: a Generator instance to use as the
        default random number generator for stochastic.
    """
    global generator  # pylint: disable=global-statement

    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be of type np.random.Generator")

    generator = rng or _default_rng


def seed(
    *,
    value: int,
) -> None:
    """Sets the seed for numpy generator.

    A new random number generator is created using
    ``numpy.random.default_rng(value)``.
    """
    global generator  # pylint: disable=global-statement

    if not isinstance(value, int):
        raise TypeError("Seed value must be an integer.")

    generator = np.random.default_rng(seed=value)
