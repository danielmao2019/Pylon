"""
Integration test for Manager.build_jobs with lightweight setup.
"""

import os
import tempfile
import json

from agents.manager.manager import Manager
from agents.manager.default_job import DefaultJob
from agents.manager.training_job import TrainingJob
from agents.manager.evaluation_job import EvaluationJob


def test_build_jobs_minimal_integration(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_root:
        # Layout: ./configs/x.py -> ./logs/x
        configs_dir = os.path.join(temp_root, "configs")
        logs_dir = os.path.join(temp_root, "logs")
        os.makedirs(configs_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # Create two configs
        cfg_train = os.path.join(configs_dir, "train_exp.py")
        cfg_eval = os.path.join(configs_dir, "eval_exp.py")

        # Minimal config contents with fields used by code
        with open(cfg_train, 'w') as f:
            f.write(
                "from runners.trainers.base_trainer import BaseTrainer\n"
                "config = { 'runner': BaseTrainer, 'epochs': 100 }\n"
            )
        with open(cfg_eval, 'w') as f:
            f.write(
                "from runners.evaluators.base_evaluator import BaseEvaluator\n"
                "config = { 'runner': BaseEvaluator }\n"
            )

        # Create matching logs dirs
        work_train = os.path.join(logs_dir, "train_exp")
        work_eval = os.path.join(logs_dir, "eval_exp")
        os.makedirs(work_train, exist_ok=True)
        os.makedirs(work_eval, exist_ok=True)

        # Trainer pattern (epoch_0 with validation_scores.json etc.)
        epoch0 = os.path.join(work_train, "epoch_0")
        os.makedirs(epoch0, exist_ok=True)
        with open(os.path.join(epoch0, "validation_scores.json"), 'w') as f:
            json.dump({"acc": 0.9}, f)
        with open(os.path.join(epoch0, "optimizer_buffer.json"), 'w') as f:
            json.dump({"lr": 1e-3}, f)
        # minimal torch file skipped (calculate_progress will count 1 epoch without loading)
        with open(os.path.join(epoch0, "training_losses.pt"), 'wb') as f:
            f.write(b"\x00")

        # Evaluator pattern
        with open(os.path.join(work_eval, "evaluation_scores.json"), 'w') as f:
            json.dump({"aggregated": {"acc": 1.0}, "per_datapoint": {"acc": [1.0]}}, f)

        # Point CWD to temp_root so BaseJob path helpers resolve
        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            # Prepare dummy system monitors (empty; not critical for detection)
            # Real SystemMonitor is required by Manager assertions; provide an empty dict instead
            monitors = {}

            commands = [
                "python main.py --config-filepath ./configs/train_exp.py",
                "python main.py --config-filepath ./configs/eval_exp.py",
            ]
            m = Manager(commands=commands, epochs=1, system_monitors=monitors)
            jobs = m.build_jobs()

            assert set(jobs.keys()) == set(commands)
            # Training job
            tjob = jobs["python main.py --config-filepath ./configs/train_exp.py"]
            assert isinstance(tjob, DefaultJob)
            assert isinstance(tjob, TrainingJob)
            assert tjob.work_dir == "./logs/train_exp"
            assert tjob.progress is not None
            # Evaluator job
            ejob = jobs["python main.py --config-filepath ./configs/eval_exp.py"]
            assert isinstance(ejob, DefaultJob)
            assert isinstance(ejob, EvaluationJob)
            assert ejob.work_dir == "./logs/eval_exp"
            assert ejob.progress is not None
        finally:
            os.chdir(cwd)


def test_build_jobs_mixed_runners_and_statuses(create_system_monitor_with_processes):
    import time

    with tempfile.TemporaryDirectory() as temp_root:
        configs_dir = os.path.join(temp_root, "configs")
        logs_dir = os.path.join(temp_root, "logs")
        os.makedirs(configs_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # trainer_running: partial epochs + fresh log
        cfg_run = os.path.join(configs_dir, 'trainer_running.py')
        with open(cfg_run, 'w') as f:
            f.write(
                "from runners.trainers.base_trainer import BaseTrainer\n"
                "config = { 'runner': BaseTrainer, 'epochs': 10 }\n"
            )
        work_run = os.path.join(logs_dir, 'trainer_running')
        os.makedirs(work_run, exist_ok=True)
        e0 = os.path.join(work_run, 'epoch_0')
        os.makedirs(e0, exist_ok=True)
        with open(os.path.join(e0, 'validation_scores.json'), 'w') as f:
            json.dump({'acc': 0.9}, f)
        with open(os.path.join(e0, 'optimizer_buffer.json'), 'w') as f:
            json.dump({'lr': 1e-3}, f)
        with open(os.path.join(e0, 'training_losses.pt'), 'wb') as f:
            f.write(b"\x00")
        # fresh log
        with open(os.path.join(work_run, 'train_val.log'), 'a'):
            os.utime(os.path.join(work_run, 'train_val.log'), None)

        # trainer_stuck: partial epochs + process mapping, no fresh log
        cfg_stuck = os.path.join(configs_dir, 'trainer_stuck.py')
        with open(cfg_stuck, 'w') as f:
            f.write(
                "from runners.trainers.base_trainer import BaseTrainer\n"
                "config = { 'runner': BaseTrainer, 'epochs': 10 }\n"
            )
        work_stuck = os.path.join(logs_dir, 'trainer_stuck')
        os.makedirs(work_stuck, exist_ok=True)
        e0s = os.path.join(work_stuck, 'epoch_0')
        os.makedirs(e0s, exist_ok=True)
        with open(os.path.join(e0s, 'validation_scores.json'), 'w') as f:
            json.dump({'acc': 0.8}, f)
        with open(os.path.join(e0s, 'optimizer_buffer.json'), 'w') as f:
            json.dump({'lr': 1e-3}, f)
        with open(os.path.join(e0s, 'training_losses.pt'), 'wb') as f:
            f.write(b"\x00")

        # evaluator_outdated: eval file aged
        cfg_eval = os.path.join(configs_dir, 'evaluator_old.py')
        with open(cfg_eval, 'w') as f:
            f.write(
                "from runners.evaluators.base_evaluator import BaseEvaluator\n"
                "config = { 'runner': BaseEvaluator }\n"
            )
        work_eval = os.path.join(logs_dir, 'evaluator_old')
        os.makedirs(work_eval, exist_ok=True)
        eval_file = os.path.join(work_eval, 'evaluation_scores.json')
        with open(eval_file, 'w') as f:
            json.dump({'aggregated': {'acc': 1.0}, 'per_datapoint': {'acc': [1.0]}}, f)
        old = time.time() - (31 * 24 * 60 * 60)
        os.utime(eval_file, (old, old))

        # CWD
        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            monitors = create_system_monitor_with_processes(
                ['python main.py --config-filepath ./configs/trainer_stuck.py']
            )
            commands = [
                "python main.py --config-filepath ./configs/trainer_running.py",
                "python main.py --config-filepath ./configs/trainer_stuck.py",
                "python main.py --config-filepath ./configs/evaluator_old.py",
            ]
            m = Manager(
                commands=commands,
                epochs=10,
                system_monitors=monitors,
                sleep_time=3600,
                outdated_days=30,
            )
            jobs = m.build_jobs()
            jr = jobs["python main.py --config-filepath ./configs/trainer_running.py"]
            js = jobs["python main.py --config-filepath ./configs/trainer_stuck.py"]
            je = jobs["python main.py --config-filepath ./configs/evaluator_old.py"]
            assert jr.status == 'running'
            assert js.status == 'stuck'
            # Evaluator completion is binary, but aged artifacts mark it as outdated
            assert je.status == 'outdated'
        finally:
            os.chdir(cwd)
