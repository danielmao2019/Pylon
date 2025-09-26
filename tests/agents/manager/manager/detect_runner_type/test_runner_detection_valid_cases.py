"""
Runner detection valid cases for Manager._detect_runner_type.
"""
import os
import tempfile
import json

from agents.manager.manager import Manager


def test_detect_evaluator_pattern():
    with tempfile.TemporaryDirectory() as temp_root:
        os.chdir(temp_root)
        os.makedirs('configs', exist_ok=True)
        os.makedirs('logs/exp', exist_ok=True)
        with open('configs/exp.py', 'w') as f:
            f.write("config = {}\n")
        with open(os.path.join('logs/exp', "evaluation_scores.json"), 'w') as f:
            json.dump({"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}, f)
        m = Manager(commands=["python main.py --config-filepath ./configs/exp.py"], epochs=1, system_monitors={})
        assert m._detect_runner_type("python main.py --config-filepath ./configs/exp.py") == 'evaluator'


def test_detect_trainer_pattern(create_epoch_files):
    with tempfile.TemporaryDirectory() as temp_root:
        os.chdir(temp_root)
        os.makedirs('configs', exist_ok=True)
        os.makedirs('logs/exp', exist_ok=True)
        with open('configs/exp.py', 'w') as f:
            f.write("from runners.trainers.base_trainer import BaseTrainer\nconfig = {'runner': BaseTrainer}\n")
        create_epoch_files('logs/exp', 0)
        m = Manager(commands=["python main.py --config-filepath ./configs/exp.py"], epochs=1, system_monitors={})
        assert m._detect_runner_type("python main.py --config-filepath ./configs/exp.py") == 'trainer'


def test_precedence_evaluator_over_trainer(create_epoch_files):
    with tempfile.TemporaryDirectory() as temp_root:
        os.chdir(temp_root)
        os.makedirs('configs', exist_ok=True)
        os.makedirs('logs/exp', exist_ok=True)
        with open('configs/exp.py', 'w') as f:
            f.write("config = {}\n")
        create_epoch_files('logs/exp', 0)
        with open(os.path.join('logs/exp', "evaluation_scores.json"), 'w') as f:
            json.dump({"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}, f)
        m = Manager(commands=["python main.py --config-filepath ./configs/exp.py"], epochs=1, system_monitors={})
        assert m._detect_runner_type("python main.py --config-filepath ./configs/exp.py") == 'evaluator'


def test_detect_with_config_runner_class():
    with tempfile.TemporaryDirectory() as temp_root:
        os.chdir(temp_root)
        os.makedirs('configs', exist_ok=True)
        os.makedirs('logs/exp', exist_ok=True)
        # evaluator
        with open('configs/exp.py', 'w') as f:
            f.write("from runners.evaluators.base_evaluator import BaseEvaluator\nconfig = { 'runner': BaseEvaluator }\n")
        m = Manager(commands=["python main.py --config-filepath ./configs/exp.py"], epochs=1, system_monitors={})
        assert m._detect_runner_type("python main.py --config-filepath ./configs/exp.py") == 'evaluator'
        # trainer
        with open('configs/exp.py', 'w') as f:
            f.write("from runners.trainers.base_trainer import BaseTrainer\nconfig = { 'runner': BaseTrainer }\n")
        assert m._detect_runner_type("python main.py --config-filepath ./configs/exp.py") == 'trainer'


def test_deterministic_detection():
    with tempfile.TemporaryDirectory() as temp_root:
        os.chdir(temp_root)
        os.makedirs('configs', exist_ok=True)
        os.makedirs('logs/exp', exist_ok=True)
        with open('configs/exp.py', 'w') as f:
            f.write("config = {}\n")
        with open(os.path.join('logs/exp', "evaluation_scores.json"), 'w') as f:
            json.dump({"aggregated": {}, "per_datapoint": {}}, f)
        m = Manager(commands=["python main.py --config-filepath ./configs/exp.py"], epochs=1, system_monitors={})
        results = [m._detect_runner_type("python main.py --config-filepath ./configs/exp.py") for _ in range(5)]
        assert all(r == 'evaluator' for r in results)
