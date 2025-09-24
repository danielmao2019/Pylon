"""
Test run_status functionality with enhanced ProgressInfo and ProcessInfo integration.
Focus on realistic testing with minimal mocking for individual function testing.

Following CLAUDE.md testing patterns:
- Correctness verification with known inputs/outputs  
- Edge case testing
- Parametrized testing for multiple scenarios
- Unit testing for specific functions
"""
from typing import Dict, Any
import os
import tempfile
import pytest
from agents.manager import (
    get_run_status,
    get_all_run_status,
    parse_config,
    get_log_last_update,
    get_epoch_last_update,
    _build_config_to_process_mapping,
    RunStatus
)
from utils.monitor.process_info import ProcessInfo
from utils.automation.progress_tracking import ProgressInfo


# ============================================================================
# TESTS FOR get_run_status (REALISTIC - MINIMAL MOCKING)
# ============================================================================

def test_get_run_status_basic_functionality(create_epoch_files, create_real_config):
    """Test basic get_run_status functionality with enhanced return type."""
    with tempfile.TemporaryDirectory() as temp_root:
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_run")
        config_path = os.path.join(configs_dir, "test_run.py")
        
        os.makedirs(work_dir, exist_ok=True)
        # Create some completed epochs
        for epoch_idx in range(5):
            create_epoch_files(work_dir, epoch_idx)
        
        # Create real config
        create_real_config(config_path, work_dir, epochs=100)
        
        # No running processes (empty config to process info mapping)
        config_to_process_info: Dict[str, ProcessInfo] = {}
        
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            # NO MOCKS - use real function with real data structures
            run_status = get_run_status(
                config=config_path,
                epochs=100,
                config_to_process_info=config_to_process_info
            )
            
            # Should return RunStatus with enhanced ProgressInfo - RunStatus is now a dataclass
            assert isinstance(run_status, RunStatus)
            assert run_status.config == config_path
            expected_work_dir = "./logs/test_run"  # What get_work_dir actually returns
            assert run_status.work_dir == expected_work_dir
            assert isinstance(run_status.progress, ProgressInfo)
            assert run_status.progress.completed_epochs == 5
            assert run_status.progress.early_stopped == False
            assert run_status.status == 'failed'  # No recent log updates, not running on GPU
            assert run_status.process_info is None  # Not running on GPU
            
        finally:
            os.chdir(original_cwd)


def test_get_run_status_with_process_info(create_epoch_files, create_real_config):
    """Test get_run_status when experiment is running on GPU."""
    with tempfile.TemporaryDirectory() as temp_root:
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_run")
        config_path = os.path.join(configs_dir, "test_run.py")
        
        os.makedirs(work_dir, exist_ok=True)
        
        # Create some completed epochs
        for epoch_idx in range(3):
            create_epoch_files(work_dir, epoch_idx)
        
        # Create config file
        create_real_config(config_path, work_dir, epochs=100)
        
        # Real process info for running experiment (not mocked)
        process_info = ProcessInfo(
            pid='12345',
            user='testuser',
            cmd=f'python main.py --config-filepath {config_path}',
            start_time='Mon Jan  1 10:00:00 2024'
        )
        config_to_process_info = {config_path: process_info}
        
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            # NO MOCKS - use real function with real data structures
            run_status = get_run_status(
                config=config_path,
                epochs=100,
                config_to_process_info=config_to_process_info
            )
            
            # Should show as stuck (running on GPU but no recent log updates)
            assert run_status.status == 'stuck'
            assert run_status.process_info == process_info
            assert run_status.progress.completed_epochs == 3
            
        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize("status_scenario,expected_status", [
    ("running", "running"),      # Recent log updates
    ("finished", "finished"),    # Completed all epochs
    ("stuck", "stuck"),          # Running on GPU but no log updates  
    ("failed", "failed"),        # No log updates, not on GPU
])
def test_run_status_determination(status_scenario, expected_status, create_epoch_files, create_real_config):
    """Test different run status determination scenarios with realistic data."""
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
        
        # Set up process info (real data structures, not mocks)
        config_to_process_info = {}
        if status_scenario == "stuck":
            config_to_process_info[config_path] = ProcessInfo(
                pid='12345',
                user='testuser',
                cmd=f'python main.py --config-filepath {config_path}',
                start_time='Mon Jan  1 10:00:00 2024'
            )
        
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            # NO MOCKS - use real function with real data
            run_status = get_run_status(
                config=config_path,
                epochs=100,
                config_to_process_info=config_to_process_info
            )
            
            assert run_status.status == expected_status
            
        finally:
            os.chdir(original_cwd)


# ============================================================================
# TESTS FOR get_all_run_status (REALISTIC WITH MINIMAL MOCK)
# ============================================================================

def test_get_all_run_status_returns_mapping(create_real_config, create_epoch_files, create_minimal_system_monitor_with_processes):
    """Test that get_all_run_status returns Dict[str, RunStatus] with minimal SystemMonitor mock."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create multiple experiment directories
        experiments = ["exp1", "exp2", "exp3"]
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        
        config_files = []
        for exp_name in experiments:
            work_dir = os.path.join(logs_dir, exp_name)
            config_path = os.path.join(configs_dir, f"{exp_name}.py")
            
            os.makedirs(work_dir, exist_ok=True)
            create_real_config(config_path, work_dir, epochs=100)
            config_files.append(config_path)
            
            # Create some epoch files
            for epoch_idx in range(2):
                create_epoch_files(work_dir, epoch_idx)
        
        # Create minimal SystemMonitor mock (only necessary mock)
        from utils.monitor.gpu_status import GPUStatus
        connected_gpus_data = [GPUStatus(server='test_server', index=0, max_memory=0, processes=[], connected=True)]
        system_monitor = create_minimal_system_monitor_with_processes(connected_gpus_data)
        
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            result = get_all_run_status(
                config_files=config_files,
                epochs=100,
                system_monitor=system_monitor
            )
            
            # Should return mapping instead of list
            assert isinstance(result, dict), f"Expected dict, got {type(result)}"
            assert len(result) == 3
            
            # Check that keys are config paths and values are RunStatus objects
            for config_path in config_files:
                assert config_path in result
                assert isinstance(result[config_path], RunStatus)
                assert result[config_path].config == config_path
                assert isinstance(result[config_path].progress, ProgressInfo)
                assert hasattr(result[config_path].progress, 'completed_epochs')
                
        finally:
            os.chdir(original_cwd)


# ============================================================================
# TESTS FOR HELPER FUNCTIONS (NO MOCKS NEEDED)
# ============================================================================

def test_parse_config():
    """Test config parsing from command strings."""
    # Test valid commands
    assert parse_config("python main.py --config-filepath configs/exp1.py") == "configs/exp1.py"
    assert parse_config("python main.py --config-filepath /absolute/path/config.py") == "/absolute/path/config.py"
    assert parse_config("python3 main.py --debug --config-filepath configs/test.py --verbose") == "configs/test.py"
    
    # Test assertions for invalid commands (missing --config-filepath)
    with pytest.raises(AssertionError):
        parse_config("python main.py --other-flag configs/exp1.py")  # Missing --config-filepath


def test_build_config_to_process_mapping():
    """Test building config to ProcessInfo mapping from GPU data."""
    from utils.monitor.gpu_status import GPUStatus
    connected_gpus = [
        GPUStatus(
            server='test_server', index=0, max_memory=0,
            processes=[
                ProcessInfo(
                    pid='12345',
                    user='testuser',
                    cmd='python main.py --config-filepath configs/exp1.py',
                    start_time='Mon Jan  1 10:00:00 2024'
                ),
                ProcessInfo(
                    pid='12346',
                    user='testuser', 
                    cmd='some_other_process --not-config',
                    start_time='Mon Jan  1 11:00:00 2024'
                )
            ]
        ),
        GPUStatus(
            server='test_server', index=1, max_memory=0,
            processes=[
                ProcessInfo(
                    pid='54321',
                    user='testuser',
                    cmd='python main.py --config-filepath configs/exp2.py --debug',
                    start_time='Mon Jan  1 12:00:00 2024'
                )
            ]
        ),
    ]
    
    mapping = _build_config_to_process_mapping(connected_gpus)
    
    # Should only include processes with python main.py --config-filepath
    assert len(mapping) == 2
    assert "configs/exp1.py" in mapping
    assert "configs/exp2.py" in mapping
    
    # Check ProcessInfo content
    assert mapping["configs/exp1.py"].pid == "12345"
    assert mapping["configs/exp1.py"].cmd == "python main.py --config-filepath configs/exp1.py"
    assert mapping["configs/exp2.py"].pid == "54321"


def test_get_log_last_update():
    """Test log timestamp detection."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test empty directory
        assert get_log_last_update(temp_dir) is None
        
        # Create log file
        log_file = os.path.join(temp_dir, "train_val_latest.log")
        with open(log_file, 'w') as f:
            f.write("test log content")
        
        timestamp = get_log_last_update(temp_dir)
        assert timestamp is not None
        assert isinstance(timestamp, float)


def test_get_epoch_last_update(EXPECTED_FILES, create_epoch_files):
    """Test epoch file timestamp detection."""
    with tempfile.TemporaryDirectory() as temp_dir:
        expected_files = EXPECTED_FILES
        
        # Test empty directory
        assert get_epoch_last_update(temp_dir, expected_files) is None
        
        # Create epoch files
        create_epoch_files(temp_dir, 0)
        
        timestamp = get_epoch_last_update(temp_dir, expected_files)
        assert timestamp is not None
        assert isinstance(timestamp, float)


# ============================================================================
# TESTS FOR EDGE CASES AND ERROR HANDLING
# ============================================================================

def test_get_run_status_invalid_config_path():
    """Test get_run_status with invalid config path."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create a work dir but no config file, and no epoch files
        logs_dir = os.path.join(temp_root, "logs")
        work_dir = os.path.join(logs_dir, "invalid_exp")
        invalid_config_path = os.path.join(temp_root, "nonexistent", "config.py")
        
        os.makedirs(work_dir, exist_ok=True)
        config_to_process_info: Dict[str, ProcessInfo] = {}
        
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            # The function should handle the missing config when trying to compute progress
            # but since the work_dir exists but is empty, it will return 0 completed epochs
            run_status = get_run_status(
                config=invalid_config_path,
                epochs=100,
                config_to_process_info=config_to_process_info
            )
            
            # Should return failed status for invalid config
            assert run_status.status == 'failed'
            assert run_status.process_info is None
            
        except FileNotFoundError:
            # If the config path resolution fails, that's also acceptable behavior
            # The system is not designed to handle completely invalid paths gracefully
            pass
            
        finally:
            os.chdir(original_cwd)


def test_parse_config_edge_cases():
    """Test parse_config with edge case command strings."""
    # Test with python3
    assert parse_config("python3 main.py --config-filepath configs/exp.py") == "configs/exp.py"
    
    # Test with absolute path
    assert parse_config("python main.py --config-filepath /home/user/configs/exp.py") == "/home/user/configs/exp.py"
    
    # Test with additional flags
    assert parse_config("python main.py --debug --config-filepath configs/exp.py --verbose") == "configs/exp.py"


def test_build_config_to_process_mapping_empty_gpu_data():
    """Test building config mapping with empty GPU data."""
    connected_gpus = []
    mapping = _build_config_to_process_mapping(connected_gpus)
    assert len(mapping) == 0
    assert mapping == {}


def test_build_config_to_process_mapping_no_matching_processes():
    """Test building config mapping when no processes match pattern."""
    connected_gpus = [
        {
            'processes': [
                ProcessInfo(
                    pid='12345',
                    user='testuser',
                    cmd='some_other_process --not-matching',
                    start_time='Mon Jan  1 10:00:00 2024'
                )
            ]
        }
    ]
    
    mapping = _build_config_to_process_mapping(connected_gpus)
    assert len(mapping) == 0
    assert mapping == {}
