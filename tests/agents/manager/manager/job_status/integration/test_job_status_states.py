"""
Job status determination tests using BaseJob utility methods and Manager.populate logic.
"""
import os
import tempfile
import time
import json

import pytest

from agents.manager.manager import Manager


def _touch(path):
    with open(path, 'a'):
        os.utime(path, None)


def test_job_status_running_vs_finished(create_system_monitor_with_processes):
    with tempfile.TemporaryDirectory() as temp_root:
        configs = os.path.join(temp_root, 'configs')
        logs = os.path.join(temp_root, 'logs')
        os.makedirs(configs, exist_ok=True)
        os.makedirs(logs, exist_ok=True)

        cfg = os.path.join(configs, 'exp.py')
        with open(cfg, 'w') as f:
            f.write(
                "from runners.trainers.base_trainer import BaseTrainer\n"
                "config = { 'runner': BaseTrainer, 'epochs': 1 }\n"
            )

        work = os.path.join(logs, 'exp')
        os.makedirs(work, exist_ok=True)
        # Trainer pattern, completed one epoch
        e0 = os.path.join(work, 'epoch_0')
        os.makedirs(e0, exist_ok=True)
        with open(os.path.join(e0, 'validation_scores.json'), 'w') as f:
            json.dump({"acc": 0.9}, f)
        with open(os.path.join(e0, 'optimizer_buffer.json'), 'w') as f:
            json.dump({"lr": 1e-3}, f)
        with open(os.path.join(e0, 'training_losses.pt'), 'wb') as f:
            f.write(b"\x00")

        # Create a recent log file to simulate running
        log_path = os.path.join(work, 'train_val.log')
        _touch(log_path)

        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            monitors = create_system_monitor_with_processes([
                'python main.py --config-filepath ./configs/exp.py'
            ])
            m = Manager(commands=["python main.py --config-filepath ./configs/exp.py"], epochs=1, system_monitors=monitors, sleep_time=3600)
            jobs = m.build_jobs()
            job = jobs["python main.py --config-filepath ./configs/exp.py"]
            # Since recent log exists and epochs=1 (complete), status should be 'running' due to recent log
            assert job.status == 'running'

            # Make log old to simulate finished state
            old = time.time() - (2 * 3600)
            os.utime(log_path, (old, old))
            jobs = m.build_jobs()
            job = jobs["python main.py --config-filepath ./configs/exp.py"]
            assert job.status in {'finished', 'outdated'}
        finally:
            os.chdir(cwd)
