from pystochastic.processes.diffusion import ExtendedVasicekProcess as TestType


def test_extended_vasicek_process_str_repr() -> None:
    instance = TestType(speed=1.0, mean=0.5, vol=0.2, t=2.0)

    assert "Extended Vasicek process with speed=" in str(instance)
    assert "ExtendedVasicekProcess(speed=" in repr(instance)
