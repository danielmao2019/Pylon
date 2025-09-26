import os
import tempfile
from agents.manager.training_job import TrainingJob
from agents.manager.runtime import JobRuntimeParams
from agents.monitor.process_info import ProcessInfo


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
        command = f"python main.py --config-filepath {config_path}"
        process_info = ProcessInfo(
            pid='12345',
            user='testuser',
            cmd=command,
            start_time='Mon Jan  1 10:00:00 2024'
        )
        config_to_process_info = {command: process_info}
        
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            # NO MOCKS - use real function with real data structures
            job_status = TrainingJob(command)
            job_status.configure(JobRuntimeParams(epochs=100, sleep_time=86400, outdated_days=30, command_processes=config_to_process_info, force_progress_recompute=False))
            
            # Should show as stuck (running on GPU but no recent log updates)
            assert job_status.status == 'stuck'
            assert job_status.process_info == process_info
            assert job_status.progress.completed_epochs == 3
            
        finally:
            os.chdir(original_cwd)
