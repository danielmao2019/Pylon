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
from utils.automation.progress_tracking.trainer_progress_tracker import TrainerProgressTracker


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


def test_trainer_progress_tracker_malformed_progress_json():
    """Test that malformed progress.json fails fast and loud."""
    with tempfile.TemporaryDirectory() as work_dir:
        config = {'epochs': 100, 'model': 'test_model'}
        
        # Create malformed progress.json
        progress_file = os.path.join(work_dir, "progress.json")
        with open(progress_file, 'w') as f:
            f.write("invalid json content {")  # Malformed JSON
        
        tracker = TrainerProgressTracker(work_dir, config)
        
        # Should fail fast and loud with JSONDecodeError when trying to read malformed JSON
        with pytest.raises(json.JSONDecodeError):
            tracker.get_progress()