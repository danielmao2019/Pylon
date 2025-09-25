"""
Runner detection valid cases for Manager._detect_runner_type.
"""
import os
import tempfile
import json

from agents.manager.manager import Manager


def test_detect_evaluator_pattern():
    with tempfile.TemporaryDirectory() as work_dir:
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump({"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}, f)
        m = Manager(config_files=[], epochs=1, system_monitors={})
        assert m._detect_runner_type(work_dir, None) == 'evaluator'


def test_detect_trainer_pattern(create_epoch_files):
    with tempfile.TemporaryDirectory() as work_dir:
        create_epoch_files(work_dir, 0)
        m = Manager(config_files=[], epochs=1, system_monitors={})
        assert m._detect_runner_type(work_dir, None) == 'trainer'


def test_precedence_evaluator_over_trainer(create_epoch_files):
    with tempfile.TemporaryDirectory() as work_dir:
        create_epoch_files(work_dir, 0)
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump({"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}, f)
        m = Manager(config_files=[], epochs=1, system_monitors={})
        assert m._detect_runner_type(work_dir, None) == 'evaluator'


def test_detect_with_config_runner_class():
    with tempfile.TemporaryDirectory() as work_dir:
        evaluator_cls = type('BaseEvaluator', (), {})
        trainer_cls = type('BaseTrainer', (), {})
        m = Manager(config_files=[], epochs=1, system_monitors={})
        assert m._detect_runner_type(work_dir, {'runner': evaluator_cls}) == 'evaluator'
        assert m._detect_runner_type(work_dir, {'runner': trainer_cls}) == 'trainer'


def test_deterministic_detection():
    with tempfile.TemporaryDirectory() as work_dir:
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump({"aggregated": {}, "per_datapoint": {}}, f)
        m = Manager(config_files=[], epochs=1, system_monitors={})
        results = [m._detect_runner_type(work_dir, None) for _ in range(5)]
        assert all(r == 'evaluator' for r in results)

