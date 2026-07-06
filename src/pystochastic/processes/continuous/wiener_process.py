from numpy.random import Generator

from .brownian_motion import BrownianMotion


class WienerProcess(BrownianMotion):
    """Wiener process, or standard Brownian motion.

    .. image:: _static/wiener_process.png
        :scale: 50%

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
        return f"Wiener process on [0, {self.t}]"

    def __repr__(self) -> str:
        return f"WienerProcess(t={self.t})"
