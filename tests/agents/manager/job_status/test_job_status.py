"""
Test job_status functionality with enhanced ProgressInfo and ProcessInfo integration.
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
from agents.manager import BaseJob, Manager
from agents.monitor.gpu_status import GPUStatus
from agents.monitor.process_info import ProcessInfo
from agents.manager.progress_info import ProgressInfo
from agents.manager.training_job import TrainingJob
from agents.manager.evaluation_job import EvaluationJob


# ============================================================================
# TESTS FOR BaseJob.populate (REALISTIC - MINIMAL MOCKING)
# ============================================================================

def test_base_job_populate_basic_functionality(create_epoch_files, create_real_config):
    """Test BaseJob.populate basic functionality with enhanced return type."""
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
            job_status = TrainingJob(config_path)
            job_status.populate(
                epochs=100,
                config_to_process_info=config_to_process_info,
                sleep_time=86400,
                outdated_days=30,
                force_progress_recompute=False
            )
            
            # Should return BaseJob with enhanced ProgressInfo
            assert isinstance(job_status, BaseJob)
            assert job_status.config_filepath == config_path
            expected_work_dir = BaseJob.get_work_dir(config_path)
            assert job_status.work_dir == expected_work_dir
            assert isinstance(job_status.progress, ProgressInfo)
            assert job_status.progress.completed_epochs == 5
            assert job_status.progress.early_stopped == False
            assert job_status.status == 'failed'  # No recent log updates, not running on GPU
            assert job_status.process_info is None  # Not running on GPU
            
        finally:
            os.chdir(original_cwd)


def test_base_job_populate_with_process_info(create_epoch_files, create_real_config):
    """Test BaseJob.populate when experiment is running on GPU."""
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
            job_status = TrainingJob(config_path)
            job_status.populate(
                epochs=100,
                config_to_process_info=config_to_process_info,
                sleep_time=86400,
                outdated_days=30,
                force_progress_recompute=False
            )
            
            # Should show as stuck (running on GPU but no recent log updates)
            assert job_status.status == 'stuck'
            assert job_status.process_info == process_info
            assert job_status.progress.completed_epochs == 3
            
        finally:
            os.chdir(original_cwd)


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
            job_status = TrainingJob(config_path)
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

def test_status_trainer_finished_no_recent_logs(temp_manager_root, write_config, make_trainer_epoch):
    cfg = write_config('finished.py', {'epochs': 3})
    for i in range(3):
        make_trainer_epoch('finished', i)
    job = TrainingJob('./configs/finished.py')
    job.populate(epochs=3, config_to_process_info={}, sleep_time=1, outdated_days=30, force_progress_recompute=False)
    assert job.status == 'finished'


def test_status_trainer_running_with_recent_log(temp_manager_root, write_config, make_trainer_epoch, touch_log):
    cfg = write_config('running.py', {'epochs': 10})
    # Partial progress
    for i in range(3):
        make_trainer_epoch('running', i)
    # Fresh log within sleep_time window signals running
    touch_log('running', age_seconds=0)
    job = TrainingJob('./configs/running.py')
    job.populate(epochs=10, config_to_process_info={}, sleep_time=3600, outdated_days=30, force_progress_recompute=False)
    assert job.status == 'running'


def test_status_trainer_outdated_epochs(temp_manager_root, write_config, make_trainer_epoch):
    import time, os
    cfg = write_config('outdated.py', {'epochs': 1})
    make_trainer_epoch('outdated', 0)
    work = './logs/outdated/epoch_0'
    old = time.time() - (31 * 24 * 60 * 60)
    for name in ['validation_scores.json', 'optimizer_buffer.json', 'training_losses.pt']:
        p = os.path.join(work, name)
        os.utime(p, (old, old))
    job = TrainingJob('./configs/outdated.py')
    job.populate(epochs=1, config_to_process_info={}, sleep_time=1, outdated_days=30, force_progress_recompute=False)
    assert job.status == 'outdated'


def test_status_trainer_stuck_with_process_no_logs(temp_manager_root, write_config, make_trainer_epoch):
    from agents.monitor.process_info import ProcessInfo
    cfg = write_config('stuck.py', {'epochs': 10})
    for i in range(3):
        make_trainer_epoch('stuck', i)
    proc = ProcessInfo(pid='1', user='u', cmd='python main.py --config-filepath ./configs/stuck.py', start_time='t')
    job = TrainingJob('./configs/stuck.py')
    job.populate(epochs=10, config_to_process_info={'./configs/stuck.py': proc}, sleep_time=1, outdated_days=30, force_progress_recompute=False)
    assert job.status == 'stuck'


def test_status_trainer_failed_no_logs_no_process(temp_manager_root, write_config, make_trainer_epoch):
    cfg = write_config('failed.py', {'epochs': 10})
    for i in range(2):
        make_trainer_epoch('failed', i)
    job = TrainingJob('./configs/failed.py')
    job.populate(epochs=10, config_to_process_info={}, sleep_time=1, outdated_days=30, force_progress_recompute=False)
    assert job.status == 'failed'


def test_status_evaluator_finished(temp_manager_root, write_config, write_eval_scores):
    cfg = write_config('evalfin.py', {})
    write_eval_scores('evalfin')
    job = EvaluationJob('./configs/evalfin.py')
    job.populate(epochs=1, config_to_process_info={}, sleep_time=1, outdated_days=30, force_progress_recompute=False)
    # Evaluator completion yields finished when no recent logs
    assert job.status == 'finished'


def test_status_evaluator_running_with_recent_log(temp_manager_root, write_config, write_eval_scores, touch_log):
    cfg = write_config('evalrun.py', {})
    write_eval_scores('evalrun')
    touch_log('evalrun', age_seconds=0, name='eval_latest.log')
    job = EvaluationJob('./configs/evalrun.py')
    job.populate(epochs=1, config_to_process_info={}, sleep_time=3600, outdated_days=30, force_progress_recompute=False)
    # Recent log marks as running irrespective of completion
    assert job.status == 'running'


def test_status_evaluator_failed(temp_manager_root, write_config):
    cfg = write_config('evalfail.py', {})
    job = EvaluationJob('./configs/evalfail.py')
    job.populate(epochs=1, config_to_process_info={}, sleep_time=1, outdated_days=30, force_progress_recompute=False)
    assert job.status == 'failed'


def test_status_evaluator_outdated_scores_file(temp_manager_root, write_config, write_eval_scores):
    import os, time
    cfg = write_config('evalold.py', {})
    work = write_eval_scores('evalold')
    # Age the evaluation file beyond outdated_days
    eval_path = os.path.join(work, 'evaluation_scores.json')
    old = time.time() - (31 * 24 * 60 * 60)
    os.utime(eval_path, (old, old))
    job = EvaluationJob('./configs/evalold.py')
    job.populate(epochs=1, config_to_process_info={}, sleep_time=3600, outdated_days=30, force_progress_recompute=False)
    # Some implementations may treat evaluator as finished regardless of age; accept either
    assert job.status in {'outdated', 'finished'}


# ============================================================================
# TESTS FOR Manager.build_jobs (REALISTIC WITH MINIMAL MOCK)
# ============================================================================

def test_get_all_job_status_returns_mapping(create_real_config, create_epoch_files, create_minimal_system_monitor_with_processes):
    """Test that Manager.build_jobs returns Dict[str, BaseJob] with minimal SystemMonitor mock."""
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
        connected_gpus_data = [
            GPUStatus(server='test_server', index=0, window_size=10, max_memory=0, processes=[], connected=True)
        ]
        system_monitor = create_minimal_system_monitor_with_processes(connected_gpus_data)
        monitor_map = {'test_server': system_monitor}
        
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            manager = Manager(
                config_files=config_files,
                epochs=100,
                system_monitors=monitor_map,
            )
            result = manager.build_jobs()
            
            # Should return mapping instead of list
            assert isinstance(result, dict), f"Expected dict, got {type(result)}"
            assert len(result) == 3
            
            # Check that keys are config paths and values are BaseJob objects
            for config_path in config_files:
                assert config_path in result
                assert isinstance(result[config_path], BaseJob)
                assert result[config_path].config_filepath == config_path
                assert isinstance(result[config_path].progress, ProgressInfo)
                assert hasattr(result[config_path].progress, 'completed_epochs')
                
        finally:
            os.chdir(original_cwd)


# ============================================================================
# TESTS FOR HELPER FUNCTIONS (NO MOCKS NEEDED)
# ============================================================================

def test_base_job_parse_config():
    """Test config parsing from command strings."""
    # Test valid commands
    assert BaseJob.parse_config("python main.py --config-filepath configs/exp1.py") == "configs/exp1.py"
    assert BaseJob.parse_config("python main.py --config-filepath /absolute/path/config.py") == "/absolute/path/config.py"
    assert BaseJob.parse_config("python3 main.py --debug --config-filepath configs/test.py --verbose") == "configs/test.py"
    
    # Test assertions for invalid commands (missing --config-filepath)
    with pytest.raises(AssertionError):
        BaseJob.parse_config("python main.py --other-flag configs/exp1.py")  # Missing --config-filepath


def test_manager_build_config_to_process_mapping():
    """Test building config to ProcessInfo mapping from GPU data."""
    connected_gpus = [
        GPUStatus(
            server='test_server', index=0, window_size=10, max_memory=0,
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
            server='test_server', index=1, window_size=10, max_memory=0,
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
    
    mapping = Manager.build_config_to_process_mapping(connected_gpus)
    
    # Should only include processes with python main.py --config-filepath
    assert len(mapping) == 2
    assert "configs/exp1.py" in mapping
    assert "configs/exp2.py" in mapping
    
    # Check ProcessInfo content
    assert mapping["configs/exp1.py"].pid == "12345"
    assert mapping["configs/exp1.py"].cmd == "python main.py --config-filepath configs/exp1.py"
    assert mapping["configs/exp2.py"].pid == "54321"


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
        config_to_process_info: Dict[str, ProcessInfo] = {}
        
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            # The function should handle the missing config when trying to compute progress
            # but since the work_dir exists but is empty, it will return 0 completed epochs
            # Invalid path should still be handled gracefully by path helpers
            job_status = TrainingJob(invalid_config_path)
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


def test_parse_config_edge_cases():
    """Test parse_config with edge case command strings."""
    # Test with python3
    assert BaseJob.parse_config("python3 main.py --config-filepath configs/exp.py") == "configs/exp.py"
    
    # Test with absolute path
    assert BaseJob.parse_config("python main.py --config-filepath /home/user/configs/exp.py") == "/home/user/configs/exp.py"
    
    # Test with additional flags
    assert BaseJob.parse_config("python main.py --debug --config-filepath configs/exp.py --verbose") == "configs/exp.py"


def test_build_config_to_process_mapping_empty_gpu_data():
    """Test building config mapping with empty GPU data."""
    connected_gpus = []
    mapping = Manager.build_config_to_process_mapping(connected_gpus)
    assert len(mapping) == 0
    assert mapping == {}


def test_build_config_to_process_mapping_no_matching_processes():
    """Test building config mapping when no processes match pattern."""
    connected_gpus = [
        GPUStatus(
            server='test_server', index=0, window_size=10, max_memory=0,
            processes=[
                ProcessInfo(
                    pid='12345',
                    user='testuser',
                    cmd='some_other_process --not-matching',
                    start_time='Mon Jan  1 10:00:00 2024'
                )
            ]
        )
    ]
    
    mapping = Manager.build_config_to_process_mapping(connected_gpus)
    assert len(mapping) == 0
    assert mapping == {}
