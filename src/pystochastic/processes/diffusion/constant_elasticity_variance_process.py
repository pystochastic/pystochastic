from typing import Callable

from numpy.random import Generator

from ...utils import ensure_single_arg_constant_function
from ...utils.validation import check_numeric
from .diffusion_process import DiffusionProcess


class ConstantElasticityVarianceProcess(DiffusionProcess):
    r"""Constant elasticity of variance process.

    .. image:: _static/constant_elasticity_variance_process.png
        :scale: 50%

    The process :math:`X_t` that satisfies the following stochastic
    differential equation with Wiener process :math:`W_t`:

    .. math::

        dX_t = \mu X_t dt + \sigma X_t^\gamma dW_t

    Realizations are generated using the Euler-Maruyama method.

    .. note::

        Since the family of diffusion processes have parameters which
        generalize to functions of ``t``, parameter attributes will be returned
        as callables, even if they are initialized as constants. e.g. a
        ``speed`` parameter of 1 accessed from an instance attribute will return
        a function which accepts a single argument and always returns 1.

    :param float drift: the drift coefficient, or :math:`\mu` above
    :param float vol: the volatility coefficient, or :math:`\sigma` above
    :param float volexp: the volatility-price exponent, or :math:`\gamma` above
    :param float t: the right hand endpoint of the time interval :math:`[0,t]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator
    """

    def __init__(
        self,
        *,
        drift: float = 1.0,
        vol: float = 1.0,
        volexp: float = 1.0,
        t: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(
            speed=-drift,
            mean=1.0,
            vol=vol,
            volexp=volexp,
            t=t,
            rng=rng,
        )

    def __str__(self) -> str:
        return (
            f"Constant elasticity of variance process with drift={self.drift}, "
            f"vol={self.vol}, volexp={self.volexp} on [0, {self.t}]"
        )

    def __repr__(self) -> str:
        return (
            f"ConstantElasticityVarianceProcess(drift={self.drift}, vol={self.vol}, "
            f"volexp={self.volexp}, t={self.t})"
        )

    @property
    def drift(self) -> Callable[[float], float]:
        """Drift, or Mu."""
        return lambda t: -self.speed(t)

    @drift.setter
    def drift(self, value: float) -> None:
        check_numeric(value=value, name="Drift coefficient.")
        self.speed = ensure_single_arg_constant_function(value=-value)
