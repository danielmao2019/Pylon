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
from agents.manager.training_job import TrainingJob


# ============================================================================
# EDGE CASE TESTS - EMPTY FILE HANDLING
# ============================================================================

def test_empty_progress_json_file_detection(create_real_config):
    """Empty progress.json should trigger recompute and yield zero progress."""
    with tempfile.TemporaryDirectory() as temp_root:
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "empty")
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

        # Create empty progress.json file
        progress_file = os.path.join(work_dir, "progress.json")
        with open(progress_file, 'w') as f:
            pass  # Create empty file

        # Verify the file exists and is empty
        assert os.path.exists(progress_file)
        assert os.path.getsize(progress_file) == 0

        # Create minimal config so TrainingJob can be instantiated
        config_path = os.path.join(configs_dir, "empty.py")
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)

        # Should raise RuntimeError from load_json about empty file
        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            progress = TrainingJob("python main.py --config-filepath ./configs/empty.py").get_progress()
        finally:
            os.chdir(cwd)

        assert progress.completed_epochs == 0
        assert progress.progress_percentage == 0.0
        assert os.path.getsize(progress_file) > 0


def test_non_empty_progress_json_loading(create_real_config):
    """Test that non-empty progress.json files are loaded correctly."""
    with tempfile.TemporaryDirectory() as temp_root:
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "non_empty")
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

        # Create valid progress.json file with content
        progress_file = os.path.join(work_dir, "progress.json")

        test_data = {
            "completed_epochs": 5,
            "progress_percentage": 50.0,
            "early_stopped": False,
            "early_stopped_at_epoch": None,
            "runner_type": "trainer",
            "total_epochs": 10,
        }

        with open(progress_file, 'w') as f:
            json.dump(test_data, f)

        # Verify the file exists and has content
        assert os.path.exists(progress_file)
        assert os.path.getsize(progress_file) > 0

        # Create minimal config so TrainingJob can be instantiated
        config_path = os.path.join(configs_dir, "non_empty.py")
        create_real_config(config_path, work_dir, epochs=10, early_stopping_enabled=False)

        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            progress = TrainingJob("python main.py --config-filepath ./configs/non_empty.py").get_progress()
        finally:
            os.chdir(cwd)

        # Verify the loaded data matches what we saved
        assert progress.completed_epochs == 5
        assert progress.progress_percentage == 50.0
        assert progress.early_stopped is False
        assert progress.early_stopped_at_epoch is None
        assert progress.runner_type == "trainer"
        assert progress.total_epochs == 10


def test_malformed_progress_json_error_handling(create_real_config):
    """Malformed progress.json should be handled by recomputing progress."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_malformed")
        config_path = os.path.join(configs_dir, "test_malformed.py")
        
        os.makedirs(work_dir, exist_ok=True)
        
        # Create malformed progress.json file
        progress_file = os.path.join(work_dir, "progress.json")
        
        with open(progress_file, 'w') as f:
            f.write("invalid json content {")  # Malformed JSON
        
        # Create real config
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)
        
        # Verify the file exists and has content
        assert os.path.exists(progress_file)
        assert os.path.getsize(progress_file) > 0
        
        expected_files = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
        
        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            progress = TrainingJob("python main.py --config-filepath ./configs/test_malformed.py").get_progress()
        finally:
            os.chdir(original_cwd)

        assert progress.completed_epochs == 0
        assert os.path.getsize(progress_file) > 0


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


def test_zero_byte_file_vs_whitespace_file(create_real_config, EXPECTED_FILES):
    """Test distinction between truly empty files and files with only whitespace."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_whitespace")
        config_path = os.path.join(configs_dir, "test_whitespace.py")
        
        os.makedirs(work_dir, exist_ok=True)
        
        # Create real config
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)
        
        # Truly empty file (0 bytes)
        empty_file = os.path.join(temp_root, "empty.json")
        with open(empty_file, 'w') as f:
            pass
        assert os.path.getsize(empty_file) == 0
        
        # File with whitespace (> 0 bytes but invalid JSON)
        whitespace_file = os.path.join(temp_root, "whitespace.json")
        with open(whitespace_file, 'w') as f:
            f.write("   \n  \t  ")
        assert os.path.getsize(whitespace_file) > 0
        
        # The empty file should be detected as size 0
        # The whitespace file should be detected as size > 0 (and would cause JSONDecodeError)
        expected_files = EXPECTED_FILES
        
        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            job = TrainingJob("python main.py --config-filepath ./configs/test_whitespace.py")
            # Empty file - should raise exception since file exists but is empty
            progress_file = os.path.join(work_dir, "progress.json")
            os.rename(empty_file, progress_file)
            
            # Should raise RuntimeError from load_json about empty file
            progress_empty = job.get_progress()
            assert progress_empty.completed_epochs == 0
            
            # Whitespace file - should raise exception since file exists but is malformed JSON
            os.rename(progress_file, empty_file)  # Move back
            os.rename(whitespace_file, progress_file)
            
            progress_whitespace = job.get_progress()
            assert progress_whitespace.completed_epochs == 0
        finally:
            os.chdir(original_cwd)
