"""
Test session_progress.py empty progress.json file handling.

Following CLAUDE.md testing patterns:
- Edge case testing pattern for empty files
- Invalid input testing with specific error verification
"""
import os
import tempfile
import json
import pytest
from utils.automation.progress_tracking.session_progress import get_session_progress
from utils.automation.progress_tracking.base_progress_tracker import ProgressInfo


# ============================================================================
# EDGE CASE TESTS - EMPTY FILE HANDLING
# ============================================================================

def test_empty_progress_json_file_detection():
    """Test that empty progress.json files are detected and skipped."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create empty progress.json file
        progress_file = os.path.join(work_dir, "progress.json")
        with open(progress_file, 'w') as f:
            pass  # Create empty file
        
        # Verify the file exists and is empty
        assert os.path.exists(progress_file)
        assert os.path.getsize(progress_file) == 0
        
        # Test that get_session_progress handles empty file correctly
        # It should detect the empty file and fall through to recomputation
        # We expect this to fail due to missing config, but importantly
        # it should NOT fail due to JSONDecodeError from trying to load empty file
        
        expected_files = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
        
        with pytest.raises(Exception) as exc_info:
            get_session_progress(work_dir, expected_files)
        
        # The error should NOT be a JSONDecodeError (which would indicate
        # we tried to load the empty file), but rather a file not found error
        # for the config file (which is expected)
        assert "JSONDecodeError" not in str(exc_info.value)
        assert "No such file or directory" in str(exc_info.value)


def test_non_empty_progress_json_loading():
    """Test that non-empty progress.json files are loaded correctly."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create valid progress.json file with content
        progress_file = os.path.join(work_dir, "progress.json")
        
        test_data = {
            "completed_epochs": 5,
            "progress_percentage": 50.0,
            "early_stopped": False,
            "early_stopped_at_epoch": None,
            "runner_type": "trainer",
            "total_epochs": 10
        }
        
        with open(progress_file, 'w') as f:
            json.dump(test_data, f)
        
        # Verify the file exists and has content
        assert os.path.exists(progress_file)
        assert os.path.getsize(progress_file) > 0
        
        # Test that get_session_progress loads the file correctly
        expected_files = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
        
        # This should successfully load from the JSON file without trying to recompute
        progress = get_session_progress(work_dir, expected_files)
        
        # Verify the loaded data matches what we saved
        assert progress.completed_epochs == 5
        assert progress.progress_percentage == 50.0
        assert progress.early_stopped == False
        assert progress.early_stopped_at_epoch is None
        assert progress.runner_type == "trainer"
        assert progress.total_epochs == 10


def test_malformed_progress_json_error_handling():
    """Test that malformed progress.json files raise appropriate errors."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create malformed progress.json file
        progress_file = os.path.join(work_dir, "progress.json")
        
        with open(progress_file, 'w') as f:
            f.write("invalid json content {")  # Malformed JSON
        
        # Verify the file exists and has content
        assert os.path.exists(progress_file)
        assert os.path.getsize(progress_file) > 0
        
        expected_files = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
        
        # This should fail with RuntimeError wrapping JSONDecodeError
        with pytest.raises(RuntimeError) as exc_info:
            get_session_progress(work_dir, expected_files)
        
        assert "Failed to load progress file" in str(exc_info.value)
        assert "JSONDecodeError" in str(exc_info.value) or "json" in str(exc_info.value).lower()


# ============================================================================
# EDGE CASE TESTS - FILE SIZE DETECTION
# ============================================================================

def test_file_size_detection_logic():
    """Test the file size detection logic works correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.json")
        
        # Test empty file
        with open(test_file, 'w') as f:
            pass  # Create empty file
        
        assert os.path.exists(test_file)
        assert os.path.getsize(test_file) == 0
        
        # Test file with content
        with open(test_file, 'w') as f:
            json.dump({"test": "data"}, f)
        
        assert os.path.getsize(test_file) > 0
        
        # Verify we can load the content
        with open(test_file, 'r') as f:
            data = json.load(f)
        assert data == {"test": "data"}


def test_zero_byte_file_vs_whitespace_file():
    """Test distinction between truly empty files and files with only whitespace."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Truly empty file (0 bytes)
        empty_file = os.path.join(temp_dir, "empty.json")
        with open(empty_file, 'w') as f:
            pass
        assert os.path.getsize(empty_file) == 0
        
        # File with whitespace (> 0 bytes but invalid JSON)
        whitespace_file = os.path.join(temp_dir, "whitespace.json")
        with open(whitespace_file, 'w') as f:
            f.write("   \n  \t  ")
        assert os.path.getsize(whitespace_file) > 0
        
        # The empty file should be detected as size 0
        # The whitespace file should be detected as size > 0 (and would cause JSONDecodeError)
        expected_files = ["training_losses.pt"]
        
        # Empty file - should skip loading and go to recomputation
        progress_file = os.path.join(temp_dir, "progress.json")
        os.rename(empty_file, progress_file)
        
        with pytest.raises(Exception) as exc_info:
            get_session_progress(temp_dir, expected_files)
        assert "JSONDecodeError" not in str(exc_info.value)  # Should not try to load
        
        # Whitespace file - should try to load and fail with JSONDecodeError
        os.rename(progress_file, empty_file)  # Move back
        os.rename(whitespace_file, progress_file)
        
        with pytest.raises(RuntimeError) as exc_info:
            get_session_progress(temp_dir, expected_files)
        assert "Failed to load progress file" in str(exc_info.value)
