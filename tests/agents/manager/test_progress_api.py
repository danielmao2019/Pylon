"""
New progress API test suite for agents.manager module.

Covers only the implemented source API:
- Manager._detect_runner_type
- TrainingJob: get_expected_files, get_log_pattern, get_session_progress, calculate_progress,
               _check_epoch_finished, _check_file_loadable
- EvaluationJob: get_expected_files, get_log_pattern, calculate_progress, _check_files_exist
"""
import os
import tempfile
import json
import torch

import pytest

from agents.manager.manager import Manager
from agents.manager.training_job import TrainingJob
from agents.manager.evaluation_job import EvaluationJob
from agents.manager.progress_info import ProgressInfo


# ---------------- Manager runner detection ----------------

def test_detect_runner_type_on_evaluator_pattern():
    with tempfile.TemporaryDirectory() as work_dir:
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump({"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}, f)
        m = Manager(config_files=[], epochs=1, system_monitors={})
        assert m._detect_runner_type(work_dir, None) == 'evaluator'


def test_detect_runner_type_on_trainer_pattern(create_epoch_files):
    with tempfile.TemporaryDirectory() as work_dir:
        create_epoch_files(work_dir, 0)
        m = Manager(config_files=[], epochs=1, system_monitors={})
        assert m._detect_runner_type(work_dir, None) == 'trainer'


def test_detect_runner_type_requires_artifacts_or_config():
    with tempfile.TemporaryDirectory() as work_dir:
        m = Manager(config_files=[], epochs=1, system_monitors={})
        with pytest.raises(ValueError):
            m._detect_runner_type(work_dir, None)


# ---------------- TrainingJob progress API ----------------

def test_trainingjob_fast_path_uses_progress_json(create_progress_json):
    with tempfile.TemporaryDirectory() as work_dir:
        create_progress_json(work_dir, completed_epochs=12, early_stopped=False, tot_epochs=100)
        progress = TrainingJob.get_session_progress(work_dir, TrainingJob.get_expected_files())
        assert isinstance(progress, ProgressInfo)
        assert progress.completed_epochs == 12
        assert progress.progress_percentage == 12.0
        assert progress.runner_type == 'trainer'


def test_trainingjob_slow_path_counts_epochs(create_epoch_files, create_real_config):
    with tempfile.TemporaryDirectory() as temp_root:
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "exp_slow")
        config_path = os.path.join(configs_dir, "exp_slow.py")
        os.makedirs(work_dir, exist_ok=True)

        for i in range(7):
            create_epoch_files(work_dir, i)

        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)

        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            progress = TrainingJob.get_session_progress(work_dir, TrainingJob.get_expected_files(), force_progress_recompute=True)
        finally:
            os.chdir(cwd)

        assert isinstance(progress, ProgressInfo)
        assert progress.completed_epochs == 7
        assert abs(progress.progress_percentage - 7.0) < 1e-9


def test_trainingjob_check_epoch_finished_and_file_loadable(tmp_path):
    epoch_dir = tmp_path / "epoch_0"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    # create expected files
    (epoch_dir / "training_losses.pt").write_bytes(b"\x80")  # will overwrite below with real tensor
    (epoch_dir / "optimizer_buffer.json").write_text(json.dumps({"lr": 1e-3}))
    (epoch_dir / "validation_scores.json").write_text(json.dumps({"acc": 0.9}))
    torch.save({"loss": torch.tensor([1.0, 0.5])}, epoch_dir / "training_losses.pt")

    assert TrainingJob._check_file_loadable(str(epoch_dir / "validation_scores.json"))
    assert TrainingJob._check_file_loadable(str(epoch_dir / "training_losses.pt"))

    assert TrainingJob._check_epoch_finished(str(epoch_dir), TrainingJob.get_expected_files())


# ---------------- EvaluationJob progress API ----------------

def test_evaluationjob_incomplete_and_complete():
    with tempfile.TemporaryDirectory() as work_dir:
        # Incomplete
        p0 = EvaluationJob.calculate_progress(work_dir, None)
        assert p0.completed_epochs == 0
        assert p0.progress_percentage == 0.0

        # Complete
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump({"aggregated": {"acc": 1.0}, "per_datapoint": {"acc": [1.0]}}, f)
        p1 = EvaluationJob.calculate_progress(work_dir, None)
        assert p1.completed_epochs == 1
        assert p1.progress_percentage == 100.0


def test_evaluationjob_check_files_exist():
    with tempfile.TemporaryDirectory() as work_dir:
        assert EvaluationJob._check_files_exist(work_dir) is False
        # empty file
        open(os.path.join(work_dir, "evaluation_scores.json"), 'w').close()
        assert EvaluationJob._check_files_exist(work_dir) is False
        # valid json
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump({"aggregated": {}, "per_datapoint": {}}, f)
        assert EvaluationJob._check_files_exist(work_dir) is True
