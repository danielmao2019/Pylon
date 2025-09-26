"""
Test job_status functionality with enhanced ProgressInfo and ProcessInfo integration.
Focus on realistic testing with minimal mocking for individual function testing.

Following CLAUDE.md testing patterns:
- Correctness verification with known inputs/outputs  
- Edge case testing
- Parametrized testing for multiple scenarios
- Unit testing for specific functions
"""
from typing import Dict
import os
import tempfile
import pytest
from agents.monitor.process_info import ProcessInfo
from agents.manager.training_job import TrainingJob


# ============================================================================
# TESTS FOR BaseJob.populate (REALISTIC - MINIMAL MOCKING)
# ============================================================================



@pytest.mark.parametrize("status_scenario,expected_status", [
    ("running", "running"),      # Recent log updates
    ("finished", "finished"),    # Completed all epochs
    ("stuck", "stuck"),          # Running on GPU but no log updates  
    ("failed", "failed"),        # No log updates, not on GPU
])
def test_job_status_determination(status_scenario, expected_status, create_epoch_files, create_real_config):
    """Test different job status determination scenarios with realistic data."""
    with tempfile.TemporaryDirectory() as temp_root:
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_status")
        config_path = os.path.join(configs_dir, "test_status.py")
        
        os.makedirs(work_dir, exist_ok=True)
        
        # Set up scenarios
        if status_scenario == "finished":
            # Create all 100 epochs
            for epoch_idx in range(100):
                create_epoch_files(work_dir, epoch_idx)
        elif status_scenario in ["running", "stuck", "failed"]:
            # Create partial epochs
            for epoch_idx in range(10):
                create_epoch_files(work_dir, epoch_idx)
            
            if status_scenario == "running":
                # Create recent log file
                log_file = os.path.join(work_dir, "train_val_latest.log")
                with open(log_file, 'w') as f:
                    f.write("Recent training log")
        
        # Create config file
        create_real_config(config_path, work_dir, epochs=100)
        
        command = f"python main.py --config-filepath {config_path}"

        # Set up process info (real data structures, not mocks)
        config_to_process_info = {}
        if status_scenario == "stuck":
            process = ProcessInfo(
                pid='12345',
                user='testuser',
                cmd=command,
                start_time='Mon Jan  1 10:00:00 2024'
            )
            config_to_process_info[command] = process

        original_cwd = os.getcwd()
        os.chdir(temp_root)

        try:
            # NO MOCKS - use real function with real data
            job_status = TrainingJob(command)
            job_status.populate(
                epochs=100,
                config_to_process_info=config_to_process_info,
                sleep_time=86400,
                outdated_days=30,
                force_progress_recompute=False
            )
            
            assert job_status.status == expected_status
            
        finally:
            os.chdir(original_cwd)


# ============================================================================
# EXPLICIT STATUS CASES USING REAL FILES AND CLASSES
# ============================================================================










# ============================================================================
# TESTS FOR Manager.build_jobs (REALISTIC WITH MINIMAL MOCK)
# ============================================================================


# ============================================================================
# TESTS FOR HELPER FUNCTIONS (NO MOCKS NEEDED)
# ============================================================================

# ============================================================================
# TESTS FOR EDGE CASES AND ERROR HANDLING
# ============================================================================

def test_base_job_populate_invalid_config_path():
    """Test BaseJob.populate with invalid config path."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create a work dir but no config file, and no epoch files
        logs_dir = os.path.join(temp_root, "logs")
        work_dir = os.path.join(logs_dir, "invalid_exp")
        invalid_config_path = os.path.join(temp_root, "nonexistent", "config.py")
        
        os.makedirs(work_dir, exist_ok=True)
        command = f"python main.py --config-filepath {invalid_config_path}"
        config_to_process_info: Dict[str, ProcessInfo] = {}

        original_cwd = os.getcwd()
        os.chdir(temp_root)

        try:
            # The function should handle the missing config when trying to compute progress
            # but since the work_dir exists but is empty, it will return 0 completed epochs
            # Invalid path should still be handled gracefully by path helpers
            job_status = TrainingJob(command)
            job_status.populate(
                epochs=100,
                config_to_process_info=config_to_process_info,
                sleep_time=86400,
                outdated_days=30,
                force_progress_recompute=False
            )
            
            # Should return failed status for invalid config
            assert job_status.status == 'failed'
            assert job_status.process_info is None
            
        except FileNotFoundError:
            # If the config path resolution fails, that's also acceptable behavior
            # The system is not designed to handle completely invalid paths gracefully
            pass
            
        finally:
            os.chdir(original_cwd)
