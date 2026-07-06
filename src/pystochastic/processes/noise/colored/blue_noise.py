from numpy.random import Generator

from .colored_noise import ColoredNoise


class BlueNoise(ColoredNoise):
    r"""Blue noise.

    .. image:: _static/blue_noise.png
        :scale: 50%

    Colored noise, or power law noise with spectral density exponent
    :math:`\beta = -1`.

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
            beta=-1,
            t=t,
            rng=rng,
        )
