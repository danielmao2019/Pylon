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
            json.dump(
                {
                    "aggregated": {"acc": 0.9},
                    "per_datapoint": {"acc": [0.8, 0.9, 0.95]},
                },
                f,
            )
        cmd = "python main.py --config-filepath ./configs/exp.py"
        m = Manager(commands=[cmd], epochs=1, system_monitors={})
        assert m._detect_from_artifacts(work_dir).name.lower() == 'evaluator'


def test_invalid_configs_errors():
    with tempfile.TemporaryDirectory() as work_dir:
        m = Manager(
            commands=["python main.py --config-filepath ./configs/exp.py"],
            epochs=1,
            system_monitors={},
        )
        # invalid config cases are now ignored by _detect_from_config and fall back
        assert m._detect_from_config(os.path.join(work_dir, 'nonexistent.py')) is None


def test_nonexistent_directory_error():
    m = Manager(
        commands=["python main.py --config-filepath /does/not/exist.py"],
        epochs=1,
        system_monitors={},
    )
    with pytest.raises(ValueError, match="Unable to determine runner type"):
        m._detect_runner_type("python main.py --config-filepath /does/not/exist.py")
