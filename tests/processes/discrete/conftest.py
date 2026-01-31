from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import pytest
import scipy.stats as ss

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


# @pytest.fixture(params=[1])
# def initial(request: "FixtureRequest[int]") -> int:
#     return request.param


# BernoulliProcess
@pytest.fixture(params=[0.5])
def p(request: "FixtureRequest[float]") -> float:
    return request.param


@pytest.fixture(params=[0.5, "0.5", 2, -0.5])
def p_fixture(request: "FixtureRequest[float]") -> float:
    return request.param


# ChineseRestaurantProcess
@pytest.fixture(params=[0])
def discount(request: "FixtureRequest[float]") -> float:
    return request.param


@pytest.fixture(params=[1])
def strength(request: "FixtureRequest[float]") -> float:
    return request.param


@pytest.fixture(params=[2, -1, 0.7])
def discount_fixture(request: "FixtureRequest[float]") -> float:
    return request.param


@pytest.fixture(params=[1.1, -2])
def strength_fixture(request: "FixtureRequest[float]") -> float:
    return request.param


# MarkovChain
@pytest.fixture(params=[[[0.25, 0.75], [0.4, 0.6]]])
def transition(request: "FixtureRequest[list[list[float]]]") -> list[list[float]]:
    return request.param


@pytest.fixture(params=[[0.25, 0.75], None])
def initial(request: "FixtureRequest[list[float] | None]") -> list[float] | None:
    return request.param


# MoranProcess
@pytest.fixture(params=[0, 0.1])
def maximum_fixture(request: "FixtureRequest[float]") -> float:
    return request.param


@pytest.fixture(params=[0, 1.1])
def n_fixture(request: "FixtureRequest[float]") -> float:
    return request.param


@pytest.fixture(params=[-1, 1.1])
def start_fixture(request: "FixtureRequest[float]") -> float:
    return request.param


@pytest.fixture(params=[1])
def start(request: "FixtureRequest[int]") -> int:
    return request.param


@pytest.fixture(params=[5])
def maximum(request: "FixtureRequest[int]") -> int:
    return request.param


# RandomWalk
@pytest.fixture(params=[[-1, 1]])
def steps(request: "FixtureRequest[list[int]]") -> list[int]:
    return request.param


@pytest.fixture(params=[[1, 1], None])
def weights(request: "FixtureRequest[list[int] | None]") -> list[int] | None:
    return request.param


@pytest.fixture(params=[[], ["1"], [[1, 2], [3, 4]]])
def steps_fixture(request: "FixtureRequest[list[int]]") -> list[int]:
    return request.param


@pytest.fixture(params=[[1], [-1, -1], [[1, 2], [3, 4]]])
def weights_fixture(request: "FixtureRequest[list[int] | None]") -> list[int] | None:
    return request.param


# Dirichlet process
@pytest.fixture(params=[None, np.random.uniform, ss.cauchy().rvs, 1])
def base(request: "FixtureRequest[float]") -> float:
    return request.param


@pytest.fixture(params=[-1, 0, 0.1, 1, 100])
def alpha(request: "FixtureRequest[float]") -> float:
    return request.param
