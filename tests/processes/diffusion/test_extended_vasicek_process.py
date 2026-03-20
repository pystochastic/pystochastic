"""Vasicek tests."""

from pystochastic.processes.diffusion import ExtendedVasicekProcess


def test_vasicek_str_repr(speed, mean, vol, t):
    instance = ExtendedVasicekProcess(speed, mean, vol, t)
    assert isinstance(repr(instance), str)
    assert isinstance(str(instance), str)
