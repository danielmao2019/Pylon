"""
Test session_progress functionality with enhanced ProgressInfo.
Focus on realistic testing with minimal mocking.

Following CLAUDE.md testing patterns:
- Correctness verification with known inputs/outputs
- Edge case testing
- Parametrized testing for multiple scenarios
- Determinism testing
"""

import os
import tempfile
import json
import pytest
import torch

from agents.manager.training_job import TrainingJob
from agents.manager.runtime import JobRuntimeParams
from agents.manager.progress_info import ProgressInfo

# ============================================================================
# TESTS FOR get_session_progress (NO MOCKS - PURE FUNCTIONS)
# ============================================================================


def test_get_session_progress_fast_path_normal_run(create_progress_json, create_real_config, EXPECTED_FILES):
    """Test fast path with progress.json for normal (non-early stopped) run."""
    with tempfile.TemporaryDirectory() as temp_root:
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "fast_normal")
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

        # Create progress.json for normal run (57/100 epochs)
        create_progress_json(work_dir, completed_epochs=57, early_stopped=False, tot_epochs=100)

        config_path = os.path.join(configs_dir, "fast_normal.py")
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)

        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            job = TrainingJob("python main.py --config-filepath ./configs/fast_normal.py")
            progress = job.compute_progress(
                JobRuntimeParams(
                    epochs=100,
                    sleep_time=1,
                    outdated_days=30,
                    command_processes={},
                    force_progress_recompute=False,
                )
            )
        finally:
            os.chdir(cwd)

        # Should return ProgressInfo dataclass
        assert isinstance(progress, ProgressInfo), f"Expected ProgressInfo, got {type(progress)}"
        assert progress.completed_epochs == 57
        assert abs(progress.progress_percentage - 57.0) < 0.01  # Handle floating point precision
        assert progress.early_stopped is False
        assert progress.early_stopped_at_epoch is None


def test_get_session_progress_fast_path_early_stopped_run(create_progress_json, create_real_config, EXPECTED_FILES):
    """Test fast path with progress.json for early stopped run."""
    with tempfile.TemporaryDirectory() as temp_root:
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "fast_early")
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

        # Create progress.json for early stopped run
        create_progress_json(
            work_dir,
            completed_epochs=57,
            early_stopped=True,
            early_stopped_at_epoch=57,
            tot_epochs=100,
        )

        config_path = os.path.join(configs_dir, "fast_early.py")
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=True)

        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            job = TrainingJob("python main.py --config-filepath ./configs/fast_early.py")
            progress = job.compute_progress(
                JobRuntimeParams(
                    epochs=100,
                    sleep_time=1,
                    outdated_days=30,
                    command_processes={},
                    force_progress_recompute=False,
                )
            )
        finally:
            os.chdir(cwd)

        # Should return ProgressInfo dataclass
        assert isinstance(progress, ProgressInfo), f"Expected ProgressInfo, got {type(progress)}"
        assert progress.completed_epochs == 57
        assert progress.progress_percentage == 100.0  # Early stopped shows 100%
        assert progress.early_stopped is True
        assert progress.early_stopped_at_epoch == 57


def test_get_session_progress_force_progress_recompute(create_progress_json, create_epoch_files, create_real_config, EXPECTED_FILES):
    """Test that force_progress_recompute bypasses cached progress.json and recomputes from filesystem."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_force_recompute")
        config_path = os.path.join(configs_dir, "test_force_recompute.py")

        os.makedirs(work_dir, exist_ok=True)
        expected_files = EXPECTED_FILES

        # Create outdated progress.json showing 2 completed epochs
        create_progress_json(work_dir, completed_epochs=2, early_stopped=False, tot_epochs=100)

        # But actually create 5 completed epochs on filesystem
        for epoch_idx in range(5):
            create_epoch_files(work_dir, epoch_idx)

        # Create real config (needed for slow path)
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)

        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)

        try:
            job = TrainingJob("python main.py --config-filepath ./configs/test_force_recompute.py")

            # Normal call should use cached progress.json
            progress_cached = job.compute_progress(
                JobRuntimeParams(
                    epochs=100,
                    sleep_time=1,
                    outdated_days=30,
                    command_processes={},
                    force_progress_recompute=False,
                )
            )
            assert progress_cached.completed_epochs == 2  # From cached progress.json
            assert progress_cached.progress_percentage == 2.0

            # Force recompute should bypass cache and recompute from filesystem
            progress_recomputed = job.compute_progress(
                JobRuntimeParams(
                    epochs=100,
                    sleep_time=1,
                    outdated_days=30,
                    command_processes={},
                    force_progress_recompute=True,
                )
            )
            assert progress_recomputed.completed_epochs == 5  # From actual filesystem
            assert progress_recomputed.progress_percentage == 5.0
            assert progress_recomputed.early_stopped == False
            assert progress_recomputed.early_stopped_at_epoch is None

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize(
    "completed_epochs,expected_completed",
    [
        (0, 0),  # No epochs completed
        (1, 1),  # One epoch completed
        (50, 50),  # Half completed
        (99, 99),  # Almost complete
        (100, 100),  # Fully complete
    ],
)
def test_get_session_progress_slow_path_normal_runs(
    completed_epochs,
    expected_completed,
    create_epoch_files,
    create_real_config,
    EXPECTED_FILES,
):
    """Test slow path (no progress.json) for various normal completion levels."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_normal_run")
        config_path = os.path.join(configs_dir, "test_normal_run.py")

        os.makedirs(work_dir, exist_ok=True)
        expected_files = EXPECTED_FILES

        # Create epoch files for completed epochs
        for epoch_idx in range(completed_epochs):
            create_epoch_files(work_dir, epoch_idx)

        # Create real config (no early stopping)
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)

        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)

        try:
            job = TrainingJob("python main.py --config-filepath ./configs/test_normal_run.py")
            progress = job.compute_progress(
                JobRuntimeParams(
                    epochs=100,
                    sleep_time=1,
                    outdated_days=30,
                    command_processes={},
                    force_progress_recompute=False,
                )
            )

            # Should return ProgressInfo dataclass
            assert isinstance(progress, ProgressInfo), f"Expected ProgressInfo, got {type(progress)}"
            assert progress.completed_epochs == expected_completed
            assert progress.early_stopped == False
            assert progress.early_stopped_at_epoch is None

            # Verify progress.json was created
            progress_file = os.path.join(work_dir, "progress.json")
            assert os.path.exists(progress_file), "progress.json should have been created"

        finally:
            os.chdir(original_cwd)


def test_get_session_progress_deterministic(create_progress_json, create_real_config, EXPECTED_FILES):
    """Test that progress calculation is deterministic across multiple calls."""
    with tempfile.TemporaryDirectory() as temp_root:
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "deterministic")
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

        create_progress_json(work_dir, completed_epochs=42, early_stopped=False)
        config_path = os.path.join(configs_dir, "deterministic.py")
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)

        results = []
        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            for _ in range(3):
                job = TrainingJob("python main.py --config-filepath ./configs/deterministic.py")
                progress = job.compute_progress(
                    JobRuntimeParams(
                        epochs=100,
                        sleep_time=1,
                        outdated_days=30,
                        command_processes={},
                        force_progress_recompute=False,
                    )
                )
                results.append(progress)
        finally:
            os.chdir(cwd)

        assert all(r == results[0] for r in results), f"Results not deterministic: {results}"
        assert results[0].completed_epochs == 42


def test_get_session_progress_edge_case_empty_work_dir(create_real_config, EXPECTED_FILES):
    """Test progress calculation with empty work directory."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_empty")
        config_path = os.path.join(configs_dir, "test_empty.py")

        os.makedirs(work_dir, exist_ok=True)
        expected_files = EXPECTED_FILES

        # Create real config
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)

        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)

        try:
            job = TrainingJob("python main.py --config-filepath ./configs/test_empty.py")
            progress = job.compute_progress(
                JobRuntimeParams(
                    epochs=1,
                    sleep_time=1,
                    outdated_days=30,
                    command_processes={},
                    force_progress_recompute=False,
                )
            )

            # Should return ProgressInfo dataclass with 0 completed epochs
            assert isinstance(progress, ProgressInfo), f"Expected ProgressInfo, got {type(progress)}"
            assert progress.completed_epochs == 0
            assert progress.early_stopped == False
            assert progress.early_stopped_at_epoch is None

        finally:
            os.chdir(original_cwd)


# ============================================================================
# TESTS FOR UTILITY FUNCTIONS
# ============================================================================


def test_check_epoch_finished(create_epoch_files, EXPECTED_FILES):
    """Test epoch completion checking with real files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        epoch_dir = os.path.join(temp_dir, "epoch_0")
        expected_files = EXPECTED_FILES

        # Test empty directory
        assert not TrainingJob._check_epoch_finished(epoch_dir, expected_files)

        # Create epoch directory
        os.makedirs(epoch_dir, exist_ok=True)
        assert not TrainingJob._check_epoch_finished(epoch_dir, expected_files)

        # Create files one by one
        create_epoch_files(temp_dir, 0)
        assert TrainingJob._check_epoch_finished(epoch_dir, expected_files)


def test_check_file_loadable():
    """Test file loading validation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test valid JSON file
        json_file = os.path.join(temp_dir, "test.json")
        with open(json_file, 'w') as f:
            json.dump({"test": "data"}, f)
        assert TrainingJob._check_file_loadable(json_file)

        # Test valid PT file
        pt_file = os.path.join(temp_dir, "test.pt")
        torch.save({"test": torch.tensor([1, 2, 3])}, pt_file)
        assert TrainingJob._check_file_loadable(pt_file)

        # Test invalid JSON file
        invalid_json = os.path.join(temp_dir, "invalid.json")
        with open(invalid_json, 'w') as f:
            f.write("invalid json content {")
        assert not TrainingJob._check_file_loadable(invalid_json)
