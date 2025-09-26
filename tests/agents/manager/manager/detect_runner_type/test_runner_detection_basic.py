"""
Core runner detection scenarios migrated from the shared progress suite.
"""
import json
import os
import tempfile

import pytest

from agents.manager.manager import Manager


def test_detect_runner_type_on_evaluator_pattern():
    with tempfile.TemporaryDirectory() as work_dir:
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump({"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}, f)
        m = Manager(commands=["python main.py --config-filepath ./configs/exp.py"], epochs=1, system_monitors={})
        assert m._detect_from_artifacts(work_dir) is not None


def test_detect_runner_type_on_trainer_pattern(create_epoch_files):
    with tempfile.TemporaryDirectory() as work_dir:
        create_epoch_files(work_dir, 0)
        m = Manager(commands=["python main.py --config-filepath ./configs/exp.py"], epochs=1, system_monitors={})
        assert m._detect_from_artifacts(work_dir) is not None


def test_detect_runner_type_requires_artifacts_or_config():
    with tempfile.TemporaryDirectory() as work_dir:
        m = Manager(commands=["python main.py --config-filepath ./configs/exp.py"], epochs=1, system_monitors={})
        with pytest.raises(ValueError):
            m._detect_runner_type("python main.py --config-filepath ./configs/exp.py")
