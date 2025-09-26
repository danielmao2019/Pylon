"""
TrainingJob progress API scenarios migrated from the shared manager suite.

Covers:
- TrainingJob.get_progress fast-path and epoch counting
- TrainingJob._check_epoch_finished and _check_file_loadable helpers
"""
import json
import os
import tempfile

import torch

from agents.manager.training_job import TrainingJob
from agents.manager.progress_info import ProgressInfo


def test_trainingjob_fast_path_uses_progress_json(create_progress_json):
    with tempfile.TemporaryDirectory() as root:
        logs_dir = os.path.join(root, 'logs')
        configs_dir = os.path.join(root, 'configs')
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

        work_dir = os.path.join(logs_dir, 'exp_fast')
        os.makedirs(work_dir, exist_ok=True)
        create_progress_json(work_dir, completed_epochs=12, early_stopped=False, tot_epochs=100)

        config_path = os.path.join(configs_dir, 'exp_fast.py')
        with open(config_path, 'w') as f:
            f.write('config = {"epochs": 100, "work_dir": "' + work_dir.replace('\\', '/') + '"}\n')

        cwd = os.getcwd()
        os.chdir(root)
        try:
            job = TrainingJob(f"python main.py --config-filepath {config_path}")
            progress = job.get_progress()
        finally:
            os.chdir(cwd)

    assert isinstance(progress, ProgressInfo)
    assert progress.completed_epochs == 12
    assert progress.progress_percentage == 12.0
    assert progress.runner_type == 'trainer'


def test_trainingjob_slow_path_counts_epochs(create_epoch_files):
    with tempfile.TemporaryDirectory() as root:
        logs_dir = os.path.join(root, 'logs')
        configs_dir = os.path.join(root, 'configs')
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

        work_dir = os.path.join(logs_dir, 'exp_slow')
        os.makedirs(work_dir, exist_ok=True)

        for i in range(7):
            create_epoch_files(work_dir, i)

        config_path = os.path.join(configs_dir, 'exp_slow.py')
        with open(config_path, 'w') as f:
            f.write('config = {"epochs": 100, "work_dir": "' + work_dir.replace('\\', '/') + '"}\n')

        cwd = os.getcwd()
        os.chdir(root)
        try:
            job = TrainingJob(f"python main.py --config-filepath {config_path}")
            progress = job.get_progress(force_progress_recompute=True)
        finally:
            os.chdir(cwd)

    assert isinstance(progress, ProgressInfo)
    assert progress.completed_epochs == 7
    assert abs(progress.progress_percentage - 7.0) < 1e-9


def test_trainingjob_check_epoch_finished_and_file_loadable(tmp_path):
    epoch_dir = tmp_path / "epoch_0"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    # create expected files
    (epoch_dir / "training_losses.pt").write_bytes(b"\x80")  # overwritten with tensor below
    (epoch_dir / "optimizer_buffer.json").write_text(json.dumps({"lr": 1e-3}))
    (epoch_dir / "validation_scores.json").write_text(json.dumps({"acc": 0.9}))
    torch.save({"loss": torch.tensor([1.0, 0.5])}, epoch_dir / "training_losses.pt")

    assert TrainingJob._check_file_loadable(str(epoch_dir / "validation_scores.json"))
    assert TrainingJob._check_file_loadable(str(epoch_dir / "training_losses.pt"))
    assert TrainingJob._check_epoch_finished(str(epoch_dir), TrainingJob.EXPECTED_FILES)
