from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamic_model_loading.fault_resources import Resources


def fake_resources():
    r=Resources.__new__(Resources)
    r.error=None
    r.sample=Mock()
    r.stop=Mock()
    r.thread=Mock()
    r.out=Mock()
    r.dll=SimpleNamespace(nvmlShutdown=Mock())
    return r


def test_latched_check_does_not_sample_inside_generation():
    r=fake_resources()
    r.check()
    r.sample.assert_not_called()
    r.error='late sampler failure'
    with pytest.raises(RuntimeError,match='late sampler failure'):
        r.check()


def test_sampler_exit_checks_final_error_and_closes_resources():
    r=fake_resources()
    def final_sample():
        r.error='late resource breach'
    r.sample.side_effect=final_sample
    with pytest.raises(RuntimeError,match='late resource breach'):
        r.__exit__(None,None,None)
    r.thread.join.assert_called_once()
    r.out.close.assert_called_once()
    r.dll.nvmlShutdown.assert_called_once()
