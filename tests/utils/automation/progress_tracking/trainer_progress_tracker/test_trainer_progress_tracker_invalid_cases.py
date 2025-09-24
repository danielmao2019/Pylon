"""
Test TrainerProgressTracker functionality - INVALID CASES (pytest.raises).

Following CLAUDE.md testing patterns:
- Invalid input testing with exception verification
- Edge case testing
"""
import os
import tempfile
import json
import pytest
from agents.tracker.trainer_progress_tracker import TrainerProgressTracker
from agents.tracker.base_progress_tracker import ProgressInfo


# ============================================================================
# INVALID TESTS - EXPECTED FAILURES (pytest.raises)
# ============================================================================

def test_trainer_progress_tracker_nonexistent_work_dir():
    """Test initialization with nonexistent work directory."""
    nonexistent_dir = "/this/path/does/not/exist"
    
    with pytest.raises(AssertionError) as exc_info:
        TrainerProgressTracker(nonexistent_dir)
    
    assert "work_dir does not exist" in str(exc_info.value)


def test_trainer_progress_tracker_invalid_work_dir_type():
    """Test initialization with invalid work_dir type."""
    with pytest.raises(AssertionError) as exc_info:
        TrainerProgressTracker(123)  # Integer instead of string
    
    assert "work_dir must be str" in str(exc_info.value)


def test_trainer_progress_tracker_malformed_progress_json(create_real_config):
    """Test that malformed progress.json fails fast and loud."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_malformed_trainer")
        config_path = os.path.join(configs_dir, "test_malformed_trainer.py")
        
        os.makedirs(work_dir, exist_ok=True)
        
        config = {'epochs': 100, 'model': 'test_model'}
        
        # Create real config
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)
        
        # Create malformed progress.json
        progress_file = os.path.join(work_dir, "progress.json")
        with open(progress_file, 'w') as f:
            f.write("invalid json content {")  # Malformed JSON
        
        tracker = TrainerProgressTracker(work_dir, config)
        
        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            # Malformed JSON should raise exception since file exists but is malformed
            with pytest.raises(RuntimeError) as exc_info:
                tracker.get_progress()
            
            # Should raise RuntimeError about JSON loading error
            assert "Error loading JSON" in str(exc_info.value)
        finally:
            os.chdir(original_cwd)
