"""Tests for BaseJob.parse_config command parsing."""

import pytest

from agents.manager.default_job import DefaultJob


@pytest.mark.parametrize(
    "cmd,expected",
    [
        (
            "python main.py --config-filepath ./configs/foo/bar.py --other-flag 1",
            "./configs/foo/bar.py",
        ),
        ("python main.py --config-filepath configs/exp1.py", "configs/exp1.py"),
        (
            "python main.py --config-filepath /absolute/path/config.py",
            "/absolute/path/config.py",
        ),
        (
            "python3 main.py --debug --config-filepath configs/test.py --verbose",
            "configs/test.py",
        ),
    ],
)
def test_parse_config_variants(cmd, expected):
    assert DefaultJob.parse_config(cmd) == expected


@pytest.mark.parametrize(
    "cmd",
    [
        "python main.py",  # missing flag
        "python main.py --other-flag configs/exp1.py",  # wrong flag
        "bash -lc 'echo hi'",  # not a python command
    ],
)
def test_parse_config_invalid(cmd):
    with pytest.raises(AssertionError):
        DefaultJob.parse_config(cmd)
