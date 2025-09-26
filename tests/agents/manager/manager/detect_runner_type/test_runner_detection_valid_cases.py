"""
Runner detection valid cases for Manager._detect_runner_type.
"""
import os
import tempfile
import json

from agents.manager.manager import Manager
from utils.io import config as config_loader


def test_detect_evaluator_pattern():
    with tempfile.TemporaryDirectory() as temp_root:
        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            os.makedirs('configs', exist_ok=True)
            os.makedirs('logs/exp', exist_ok=True)
            config_path = os.path.abspath(os.path.join('configs', 'exp_eval.py'))
            with open(config_path, 'w') as f:
                f.write(
                    "from runners.evaluators.base_evaluator import BaseEvaluator\n"
                    "config = {'runner': BaseEvaluator}\n"
                )
            with open(os.path.join('logs/exp', "evaluation_scores.json"), 'w') as f:
                json.dump({"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}, f)
            command = f"python main.py --config-filepath {config_path}"
            m = Manager(commands=[command], epochs=1, system_monitors={})
            assert m._detect_runner_type(command) == 'evaluator'
        finally:
            os.chdir(cwd)


def test_detect_trainer_pattern(create_epoch_files):
    with tempfile.TemporaryDirectory() as temp_root:
        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            os.makedirs('configs', exist_ok=True)
            os.makedirs('logs/exp', exist_ok=True)
            config_path = os.path.abspath(os.path.join('configs', 'exp_trainer.py'))
            with open(config_path, 'w') as f:
                f.write("from runners.trainers.base_trainer import BaseTrainer\nconfig = {'runner': BaseTrainer}\n")
            create_epoch_files('logs/exp', 0)
            command = f"python main.py --config-filepath {config_path}"
            m = Manager(commands=[command], epochs=1, system_monitors={})
            assert m._detect_runner_type(command) == 'trainer'
        finally:
            os.chdir(cwd)


def test_precedence_evaluator_over_trainer(create_epoch_files):
    with tempfile.TemporaryDirectory() as temp_root:
        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            os.makedirs('configs', exist_ok=True)
            os.makedirs('logs/exp', exist_ok=True)
            config_path = os.path.abspath(os.path.join('configs', 'exp_precedence.py'))
            with open(config_path, 'w') as f:
                f.write(
                    "from runners.evaluators.base_evaluator import BaseEvaluator\n"
                    "config = {'runner': BaseEvaluator}\n"
                )
            create_epoch_files('logs/exp', 0)
            with open(os.path.join('logs/exp', "evaluation_scores.json"), 'w') as f:
                json.dump({"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}, f)
            command = f"python main.py --config-filepath {config_path}"
            m = Manager(commands=[command], epochs=1, system_monitors={})
            assert m._detect_runner_type(command) == 'evaluator'
        finally:
            os.chdir(cwd)


def test_detect_with_config_runner_class():
    with tempfile.TemporaryDirectory() as temp_root:
        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            os.makedirs('configs', exist_ok=True)
            os.makedirs('logs/exp', exist_ok=True)
            config_path = os.path.abspath(os.path.join('configs', 'exp_toggle.py'))
            command = f"python main.py --config-filepath {config_path}"
            # evaluator
            with open(config_path, 'w') as f:
                f.write("from runners.evaluators.base_evaluator import BaseEvaluator\nconfig = { 'runner': BaseEvaluator }\n")
            config_loader._config_cache.pop(config_path, None)
            m = Manager(commands=[command], epochs=1, system_monitors={})
            assert m._detect_runner_type(command) == 'evaluator'
            # trainer
            with open(config_path, 'w') as f:
                f.write("from runners.trainers.base_trainer import BaseTrainer\nconfig = { 'runner': BaseTrainer }\n")
            config_loader._config_cache.pop(config_path, None)
            assert m._detect_runner_type(command) == 'trainer'
        finally:
            os.chdir(cwd)


def test_deterministic_detection():
    with tempfile.TemporaryDirectory() as temp_root:
        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            os.makedirs('configs', exist_ok=True)
            os.makedirs('logs/exp', exist_ok=True)
            config_path = os.path.abspath(os.path.join('configs', 'exp_deterministic.py'))
            with open(config_path, 'w') as f:
                f.write(
                    "from runners.evaluators.base_evaluator import BaseEvaluator\n"
                    "config = {'runner': BaseEvaluator}\n"
                )
            with open(os.path.join('logs/exp', "evaluation_scores.json"), 'w') as f:
                json.dump({"aggregated": {}, "per_datapoint": {}}, f)
            command = f"python main.py --config-filepath {config_path}"
            m = Manager(commands=[command], epochs=1, system_monitors={})
            results = [m._detect_runner_type(command) for _ in range(5)]
            assert all(r == 'evaluator' for r in results)
        finally:
            os.chdir(cwd)
