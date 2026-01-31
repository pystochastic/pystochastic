import pytest

from pystochastic.processes.base import BaseSequenceProcess


def test_base_sequence_process_abstract() -> None:
    with pytest.raises(TypeError):
        # pylint: disable=abstract-class-instantiated
        _ = BaseSequenceProcess()  # type: ignore[abstract]
