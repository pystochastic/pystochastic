from numpy.random import Generator

from .extended_vasicek_process import ExtendedVasicekProcess


class VasicekProcess(ExtendedVasicekProcess):
    r"""Vasicek process.

    A model for instantaneous interest rate.

    .. image:: _static/vasicek_process.png
        :scale: 50%

    The Vasicek process :math:`X_t` that satisfies the following stochastic
    differential equation with Wiener process :math:`W_t`:

    .. math::

        dX_t = \theta (\mu - X_t) dt + \sigma dW_t

    Realizations are generated using the Euler-Maruyama method.

    .. note::

        Since the family of diffusion processes have parameters which
        generalize to functions of ``t``, parameter attributes will be returned
        as callables, even if they are initialized as constants. e.g. a
        ``speed`` parameter of 1 accessed from an instance attribute will
        return a function which accepts a single argument and always returns 1.

    :param float speed: the speed of reversion, or :math:`\theta` above
    :param float mean: the mean of the process, or :math:`\mu` above
    :param float vol: volatility coefficient of the process, or :math:`\sigma`
        above
    :param float t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        speed: float = 1.0,
        mean: float = 1.0,
        vol: float = 1.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(
            speed=speed,
            mean=mean,
            vol=vol,
            t=t,
            rng=rng,
        )

    def __str__(self) -> str:
        return (
            f"Vasicek process with speed={self.speed}, "
            f"mean={self.mean}, vol={self.vol} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"VasicekProcess(speed={self.speed}, " f"mean={self.mean}, vol={self.vol}, t={self.t})"
        )
