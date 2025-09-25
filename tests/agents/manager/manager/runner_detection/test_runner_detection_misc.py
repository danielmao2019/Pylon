"""
Additional runner detection coverage migrated from legacy file.
"""
import os
import tempfile
import json
import pytest

from agents.manager.manager import Manager


def test_evaluator_takes_precedence_over_trainer(create_epoch_files):
    with tempfile.TemporaryDirectory() as work_dir:
        create_epoch_files(work_dir, 0)
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump({"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.8, 0.9, 0.95]}}, f)
        m = Manager(config_files=[], epochs=1, system_monitors={})
        assert m._detect_runner_type(work_dir, None) == 'evaluator'


def test_invalid_configs_errors():
    with tempfile.TemporaryDirectory() as work_dir:
        m = Manager(config_files=[], epochs=1, system_monitors={})
        with pytest.raises(AssertionError, match="Config must have 'runner' key"):
            m._detect_runner_type(work_dir, {'epochs': 100, 'some_other_field': 'value'})
        with pytest.raises(AssertionError, match="Expected runner to be a class"):
            m._detect_runner_type(work_dir, {'runner': 'BaseEvaluator'})


def test_nonexistent_directory_error():
    m = Manager(config_files=[], epochs=1, system_monitors={})
    with pytest.raises(ValueError, match="Unable to determine runner type"):
        m._detect_runner_type("/this/path/does/not/exist", None)

