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
from utils.automation.run_status import (
    get_session_progress,
    check_epoch_finished,
    check_file_loadable
)
from conftest import (
    create_epoch_files,
    create_progress_json,
    create_real_config,
    EXPECTED_FILES
)


# ============================================================================
# TESTS FOR get_session_progress (NO MOCKS - PURE FUNCTIONS)
# ============================================================================

def test_get_session_progress_fast_path_normal_run():
    """Test fast path with progress.json for normal (non-early stopped) run."""
    with tempfile.TemporaryDirectory() as work_dir:
        expected_files = EXPECTED_FILES
        
        # Create progress.json for normal run (57/100 epochs)
        create_progress_json(work_dir, completed_epochs=57, early_stopped=False, tot_epochs=100)
        
        progress = get_session_progress(work_dir, expected_files)
        
        # Should return ProgressInfo dict
        assert isinstance(progress, dict), f"Expected dict, got {type(progress)}"
        assert progress["completed_epochs"] == 57
        assert abs(progress["progress_percentage"] - 57.0) < 0.01  # Handle floating point precision
        assert progress["early_stopped"] == False
        assert progress["early_stopped_at_epoch"] is None


def test_get_session_progress_fast_path_early_stopped_run():
    """Test fast path with progress.json for early stopped run."""
    with tempfile.TemporaryDirectory() as work_dir:
        expected_files = EXPECTED_FILES
        
        # Create progress.json for early stopped run
        create_progress_json(work_dir, completed_epochs=57, early_stopped=True, 
                           early_stopped_at_epoch=57, tot_epochs=100)
        
        progress = get_session_progress(work_dir, expected_files)
        
        # Should return ProgressInfo dict
        assert isinstance(progress, dict), f"Expected dict, got {type(progress)}"
        assert progress["completed_epochs"] == 57
        assert progress["progress_percentage"] == 100.0  # Early stopped shows 100%
        assert progress["early_stopped"] == True
        assert progress["early_stopped_at_epoch"] == 57


@pytest.mark.parametrize("completed_epochs,expected_completed", [
    (0, 0),      # No epochs completed
    (1, 1),      # One epoch completed  
    (50, 50),    # Half completed
    (99, 99),    # Almost complete
    (100, 100),  # Fully complete
])
def test_get_session_progress_slow_path_normal_runs(completed_epochs, expected_completed):
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
            progress = get_session_progress(work_dir, expected_files)
            
            # Should return ProgressInfo dict
            assert isinstance(progress, dict), f"Expected dict, got {type(progress)}"
            assert progress["completed_epochs"] == expected_completed
            assert progress["early_stopped"] == False
            assert progress["early_stopped_at_epoch"] is None
            
            # Verify progress.json was created
            progress_file = os.path.join(work_dir, "progress.json")
            assert os.path.exists(progress_file), "progress.json should have been created"
            
        finally:
            os.chdir(original_cwd)


def test_get_session_progress_deterministic():
    """Test that progress calculation is deterministic across multiple calls."""
    with tempfile.TemporaryDirectory() as work_dir:
        expected_files = EXPECTED_FILES
        
        # Create progress.json
        create_progress_json(work_dir, completed_epochs=42, early_stopped=False)
        
        # Run multiple times and verify same result
        results = []
        for _ in range(3):
            progress = get_session_progress(work_dir, expected_files)
            results.append(progress)
        
        assert all(r == results[0] for r in results), f"Results not deterministic: {results}"
        assert results[0]["completed_epochs"] == 42


def test_get_session_progress_edge_case_empty_work_dir():
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
            progress = get_session_progress(work_dir, expected_files)
            
            # Should return ProgressInfo dict with 0 completed epochs
            assert isinstance(progress, dict), f"Expected dict, got {type(progress)}"
            assert progress["completed_epochs"] == 0
            assert progress["early_stopped"] == False
            assert progress["early_stopped_at_epoch"] is None
            
        finally:
            os.chdir(original_cwd)


# ============================================================================
# TESTS FOR UTILITY FUNCTIONS
# ============================================================================

def test_check_epoch_finished():
    """Test epoch completion checking with real files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        epoch_dir = os.path.join(temp_dir, "epoch_0")
        expected_files = EXPECTED_FILES
        
        # Test empty directory
        assert not check_epoch_finished(epoch_dir, expected_files)
        
        # Create epoch directory
        os.makedirs(epoch_dir, exist_ok=True)
        assert not check_epoch_finished(epoch_dir, expected_files)
        
        # Create files one by one
        create_epoch_files(temp_dir, 0)
        assert check_epoch_finished(epoch_dir, expected_files)


def test_check_file_loadable():
    """Test file loading validation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test valid JSON file
        json_file = os.path.join(temp_dir, "test.json")
        with open(json_file, 'w') as f:
            json.dump({"test": "data"}, f)
        assert check_file_loadable(json_file)
        
        # Test valid PT file
        pt_file = os.path.join(temp_dir, "test.pt")
        torch.save({"test": torch.tensor([1, 2, 3])}, pt_file)
        assert check_file_loadable(pt_file)
        
        # Test invalid JSON file
        invalid_json = os.path.join(temp_dir, "invalid.json")
        with open(invalid_json, 'w') as f:
            f.write("invalid json content {")
        assert not check_file_loadable(invalid_json)
