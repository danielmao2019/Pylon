"""
Tests for BaseJob.parse_config command parsing.
"""
import pytest

from agents.manager.base_job import BaseJob


def test_parse_config_happy_path():
    cmd = "python main.py --config-filepath ./configs/foo/bar.py --other-flag 1"
    assert BaseJob.parse_config(cmd) == "./configs/foo/bar.py"


@pytest.mark.parametrize(
    "cmd",
    [
        "python main.py",  # missing flag
        "bash -lc 'echo hi'",  # not a python command
    ],
)
def test_parse_config_invalid(cmd):
    with pytest.raises(AssertionError):
        BaseJob.parse_config(cmd)
