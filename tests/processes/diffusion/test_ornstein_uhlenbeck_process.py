from pystochastic.processes.diffusion import OrnsteinUhlenbeckProcess as TestType


def test_ornstein_uhlenbeck_process_str_repr() -> None:
    instance = TestType(speed=1.0, vol=0.5, t=2.0)

    assert "Ornstein-Uhlenbeck process with speed=" in str(instance)
    assert "OrnsteinUhlenbeckProcess(speed=" in repr(instance)
