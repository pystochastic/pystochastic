import math
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import numpy as np
import numpy.typing as npt
import pytest

if TYPE_CHECKING:
    from pytest import FixtureRequest as __FixtureRequest

    T = TypeVar("T")

    class FixtureRequest(__FixtureRequest, Generic[T]):
        param: T

else:
    from pytest import FixtureRequest


# Floating point arithmetic comparison threshold
@pytest.fixture(params=[10**-10])
def threshold(request: "FixtureRequest[float]") -> float:
    return request.param


# Common
@pytest.fixture(params=[1])
def t(request: "FixtureRequest[float]") -> float:
    return request.param


@pytest.fixture(params=[16])
def n(request: "FixtureRequest[int]") -> int:
    return request.param


# Generate some random times for the sample_at() method
times_random = np.cumsum(np.abs(np.random.normal(size=16)))
times_random_zero = np.cumsum([0] + list(np.abs(np.random.normal(size=16))))


@pytest.fixture(params=[times_random, times_random_zero])
def times(request: "FixtureRequest[np.ndarray]") -> npt.NDArray[np.float64]:
    return request.param


# Bessel
@pytest.fixture(params=[0, 1, 1.1])
def dim_fixture(request: "FixtureRequest[float]") -> float:
    return request.param


@pytest.fixture(params=[3])
def dim(request: "FixtureRequest[float]") -> float:
    return request.param


# BrownianBridge
@pytest.fixture(params=[3, 0, None])
def b(request: "FixtureRequest[float | None]") -> float | None:
    return request.param


# BrownianMotion
@pytest.fixture(params=[0, 1])
def drift(request: "FixtureRequest[float]") -> float:
    return request.param


@pytest.fixture(params=[1])
def scale(request: "FixtureRequest[float]") -> float:
    return request.param


# FractionalBrownianMotion
@pytest.fixture(params=[0.2, 0.5, 0.7])
def hurst(request: "FixtureRequest[float]") -> float:
    return request.param


# GammaProcess
@pytest.fixture(params=[1, None])
def mean_fixture(request: "FixtureRequest[float | None]") -> float | None:
    return request.param


@pytest.fixture(params=[1, None])
def scale_fixture(request: "FixtureRequest[float | None]") -> float | None:
    return request.param


@pytest.fixture(params=[1, None])
def rate_fixture(request: "FixtureRequest[float | None]") -> float | None:
    return request.param


@pytest.fixture(params=[1, None])
def variance_fixture(request: "FixtureRequest[float | None]") -> float | None:
    return request.param


@pytest.fixture(params=[1])
def mean(request: "FixtureRequest[float]") -> float:
    return request.param


@pytest.fixture(params=[1])
def variance(request: "FixtureRequest[float]") -> float:
    return request.param


# GeometricBrownianMotion
@pytest.fixture(params=[1])
def volatility(request: "FixtureRequest[float]") -> float:
    return request.param


@pytest.fixture(params=[1])
def initial(request: "FixtureRequest[float]") -> float:
    return request.param


# InverseGaussianProcess
def mean_func_monotonic(tt: float) -> float:
    return tt


def mean_func_not_monotonic(_: float) -> float:
    return 1


def mean_func_no_args() -> float:
    return 1


@pytest.fixture(params=[mean_func_monotonic, None])
def mean_func(
    request: "FixtureRequest[Callable[[float], float] | None]",
) -> Callable[[float], float] | None:
    return request.param


@pytest.fixture(params=[mean_func_not_monotonic, mean_func_no_args, 1])
def mean_func_invalid(
    request: "FixtureRequest[Callable[[float], float]| Callable[[], float] | float]",
) -> Callable[[float], float] | Callable[[], float] | float:
    return request.param


# MultifractionalBrownianMotion
def hurst_const(_: float) -> float:
    return 0.5


def hurst_sin(tt: float) -> float:
    return math.sin(tt) / 3 + 0.5


@pytest.fixture(params=[None, hurst_const, hurst_sin])
def hurst_func(
    request: "FixtureRequest[None | Callable[[float], float]]",
) -> None | Callable[[float], float]:
    return request.param


def hurst_too_many_args(_: float, __: float) -> float:
    return 0.5


def hurst_out_of_range(_: float) -> float:
    return 1.1


@pytest.fixture(params=[0.5, hurst_too_many_args, hurst_out_of_range])
def hurst_invalid(
    request: "FixtureRequest[float | Callable[[float], float]]",
) -> float | Callable[[float], float]:
    return request.param


# PoissonProcess
@pytest.fixture(params=[16, None])
def n_fixture(request: "FixtureRequest[int | None]") -> int | None:
    return request.param


@pytest.fixture(params=[1, None])
def length(request: "FixtureRequest[float | None]") -> float | None:
    return request.param


@pytest.fixture(params=[1])
def rate(request: "FixtureRequest[float]") -> float:
    return request.param


# MixedPoissonProcess
@pytest.fixture(params=[np.random.uniform])
def rate_func(
    request: "FixtureRequest[Callable[[],float]]",
) -> Callable[[], float]:
    return request.param


@pytest.fixture(params=[(1, 100), (1, 10)])
def rate_args(
    request: "FixtureRequest[tuple[float, float]]",
) -> tuple[float, float]:
    return request.param


@pytest.fixture(params=[{"size": None}])
def rate_kwargs(
    request: "FixtureRequest[dict[str, float | None]]",
) -> dict[str, float | None]:
    return request.param


@pytest.fixture(params=[0])
def rate_func_invalid(request: "FixtureRequest[float]") -> float:
    return request.param


@pytest.fixture(params=[0])
def rate_args_invalid(request: "FixtureRequest[float]") -> float:
    return request.param


@pytest.fixture(params=[0])
def rate_kwargs_invalid(request: "FixtureRequest[float]") -> float:
    return request.param
