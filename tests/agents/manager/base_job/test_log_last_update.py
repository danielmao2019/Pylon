import os
import tempfile

from agents.manager import BaseJob


def test_base_job_get_log_last_update():
    """Test log timestamp detection."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test empty directory
        assert BaseJob.get_log_last_update(temp_dir) is None
        
        # Create log file
        log_file = os.path.join(temp_dir, "train_val_latest.log")
        with open(log_file, 'w') as f:
            f.write("test log content")
        
        timestamp = BaseJob.get_log_last_update(temp_dir)
        assert timestamp is not None
        assert isinstance(timestamp, float)


def test_base_job_get_epoch_last_update(EXPECTED_FILES, create_epoch_files):
    """Test epoch file timestamp detection."""
    with tempfile.TemporaryDirectory() as temp_dir:
        expected_files = EXPECTED_FILES
        
        # Test empty directory
        assert BaseJob.get_epoch_last_update(temp_dir, expected_files) is None
        
        # Create epoch files
        create_epoch_files(temp_dir, 0)
        
        timestamp = BaseJob.get_epoch_last_update(temp_dir, expected_files)
        assert timestamp is not None
        assert isinstance(timestamp, float)
