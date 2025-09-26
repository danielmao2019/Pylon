"""
Runner detection invalid cases for Manager._detect_runner_type.
"""
import os
import tempfile
import pytest

from agents.manager.manager import Manager


def test_fail_fast_no_patterns():
    with tempfile.TemporaryDirectory() as temp_root:
        os.chdir(temp_root)
        os.makedirs('configs', exist_ok=True)
        os.makedirs('logs/exp', exist_ok=True)
        with open('configs/exp.py', 'w') as f:
            f.write("config = {}\n")
        # no artifacts
        m = Manager(commands=[], epochs=1, system_monitors={})
        with pytest.raises(ValueError, match="Unable to determine runner type"):
            m._detect_runner_type("python main.py --config-filepath ./configs/exp.py")


def test_fail_fast_nonexistent_directory():
    m = Manager(commands=[], epochs=1, system_monitors={})
    with pytest.raises(ValueError, match="Unable to determine runner type"):
        m._detect_runner_type("python main.py --config-filepath /path/does/not/exist.py")


def test_invalid_config_runner_missing_key():
    with tempfile.TemporaryDirectory() as temp_root:
        os.chdir(temp_root)
        os.makedirs('configs', exist_ok=True)
        with open('configs/exp.py', 'w') as f:
            f.write("config = { 'epochs': 100 }\n")
        m = Manager(commands=[], epochs=1, system_monitors={})
        with pytest.raises(AssertionError, match="Config must have 'runner' key"):
            m._detect_runner_type("python main.py --config-filepath ./configs/exp.py")


def test_invalid_config_runner_not_class():
    with tempfile.TemporaryDirectory() as temp_root:
        os.chdir(temp_root)
        os.makedirs('configs', exist_ok=True)
        with open('configs/exp.py', 'w') as f:
            f.write("config = { 'runner': 'BaseEvaluator' }\n")
        m = Manager(commands=[], epochs=1, system_monitors={})
        with pytest.raises(AssertionError, match="Expected runner to be a class"):
            m._detect_runner_type("python main.py --config-filepath ./configs/exp.py")


def test_invalid_config_runner_unhelpful_class():
    with tempfile.TemporaryDirectory() as temp_root:
        os.chdir(temp_root)
        os.makedirs('configs', exist_ok=True)
        with open('configs/exp.py', 'w') as f:
            f.write("class Other: pass\nconfig = { 'runner': Other }\n")
        m = Manager(commands=[], epochs=1, system_monitors={})
        with pytest.raises(ValueError, match="Unable to determine runner type"):
            m._detect_runner_type("python main.py --config-filepath ./configs/exp.py")
