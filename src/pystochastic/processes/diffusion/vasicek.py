from numpy.random import Generator

from ...utils import single_arg_constant_function
from .extended_vasicek import ExtendedVasicekProcess


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
        speed: float | int = 1,
        mean: float | int = 1,
        vol: float | int = 1,
        t: float = 1.0,
        rng: Generator | None = None,
    ):
        super().__init__(
            speed=single_arg_constant_function(speed),
            mean=single_arg_constant_function(mean),
            vol=single_arg_constant_function(vol),
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
            f"VasicekProcess(speed={self.speed}, "
            f"mean={self.mean}, vol={self.vol}, t={self.t})"
        )
