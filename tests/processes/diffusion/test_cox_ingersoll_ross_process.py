from pystochastic.processes.diffusion import CoxIngersollRossProcess as TestType


def test_cox_ingersoll_ross_process_str_repr() -> None:
    instance = TestType(speed=1.0, mean=0.5, vol=0.2, t=2.0)

    assert "Cox-Ingersoll-Ross process with speed=" in str(instance)
    assert "CoxIngersollRossProcess(speed=" in repr(instance)
