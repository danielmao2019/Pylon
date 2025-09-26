"""
Runner detection invalid cases for Manager._detect_runner_type.
"""
import os
import tempfile
import pytest

from agents.manager.manager import Manager
from utils.io import config as config_loader


def test_fail_fast_no_patterns():
    with tempfile.TemporaryDirectory() as temp_root:
        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            os.makedirs('configs', exist_ok=True)
            os.makedirs('logs/exp', exist_ok=True)
            config_path = os.path.abspath(os.path.join('configs', 'exp_empty.py'))
            with open(config_path, 'w') as f:
                f.write("config = {}\n")
            # no artifacts
            m = Manager(commands=[], epochs=1, system_monitors={})
            with pytest.raises(AssertionError):
                m._detect_runner_type(f"python main.py --config-filepath {config_path}")
        finally:
            os.chdir(cwd)


def test_fail_fast_nonexistent_directory():
    m = Manager(commands=[], epochs=1, system_monitors={})
    with pytest.raises(ValueError, match="Unable to determine runner type"):
        m._detect_runner_type("python main.py --config-filepath /path/does/not/exist.py")


def test_invalid_config_runner_missing_key():
    with tempfile.TemporaryDirectory() as temp_root:
        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            os.makedirs('configs', exist_ok=True)
            config_path = os.path.abspath(os.path.join('configs', 'exp_missing_runner.py'))
            with open(config_path, 'w') as f:
                f.write("config = { 'epochs': 100 }\n")
            m = Manager(commands=[], epochs=1, system_monitors={})
            with pytest.raises(AssertionError):
                m._detect_runner_type(f"python main.py --config-filepath {config_path}")
        finally:
            os.chdir(cwd)


def test_invalid_config_runner_not_class():
    with tempfile.TemporaryDirectory() as temp_root:
        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            os.makedirs('configs', exist_ok=True)
            config_path = os.path.abspath(os.path.join('configs', 'exp_string_runner.py'))
            with open(config_path, 'w') as f:
                f.write("config = { 'runner': 'BaseEvaluator' }\n")
            m = Manager(commands=[], epochs=1, system_monitors={})
            with pytest.raises(AssertionError):
                m._detect_runner_type(f"python main.py --config-filepath {config_path}")
        finally:
            os.chdir(cwd)


def test_invalid_config_runner_unhelpful_class():
    with tempfile.TemporaryDirectory() as temp_root:
        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            os.makedirs('configs', exist_ok=True)
            config_path = os.path.abspath(os.path.join('configs', 'exp_other_runner.py'))
            with open(config_path, 'w') as f:
                f.write("class Other: pass\nconfig = { 'runner': Other }\n")
            config_loader._config_cache.pop(config_path, None)
            m = Manager(commands=[], epochs=1, system_monitors={})
            with pytest.raises(ValueError, match="Unable to determine runner type"):
                m._detect_runner_type(f"python main.py --config-filepath {config_path}")
        finally:
            os.chdir(cwd)
