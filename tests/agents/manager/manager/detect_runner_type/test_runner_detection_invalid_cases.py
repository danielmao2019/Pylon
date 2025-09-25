"""
Runner detection invalid cases for Manager._detect_runner_type.
"""
import os
import tempfile
import pytest

from agents.manager.manager import Manager


def test_fail_fast_no_patterns():
    with tempfile.TemporaryDirectory() as work_dir:
        with open(os.path.join(work_dir, "some_file.txt"), 'w') as f:
            f.write("irrelevant")
        m = Manager(config_files=[], epochs=1, system_monitors={})
        with pytest.raises(ValueError, match="Unable to determine runner type"):
            m._detect_runner_type(work_dir, None)


def test_fail_fast_nonexistent_directory():
    m = Manager(config_files=[], epochs=1, system_monitors={})
    with pytest.raises(ValueError, match="Unable to determine runner type"):
        m._detect_runner_type("/path/does/not/exist", None)


def test_invalid_config_runner_missing_key():
    with tempfile.TemporaryDirectory() as work_dir:
        m = Manager(config_files=[], epochs=1, system_monitors={})
        with pytest.raises(AssertionError, match="Config must have 'runner' key"):
            m._detect_runner_type(work_dir, {'epochs': 100})


def test_invalid_config_runner_not_class():
    with tempfile.TemporaryDirectory() as work_dir:
        m = Manager(config_files=[], epochs=1, system_monitors={})
        with pytest.raises(AssertionError, match="Expected runner to be a class"):
            m._detect_runner_type(work_dir, {'runner': 'BaseEvaluator'})


def test_invalid_config_runner_unhelpful_class():
    with tempfile.TemporaryDirectory() as work_dir:
        Other = type('Other', (), {})
        m = Manager(config_files=[], epochs=1, system_monitors={})
        with pytest.raises(ValueError, match="Unable to determine runner type"):
            m._detect_runner_type(work_dir, {'runner': Other})

