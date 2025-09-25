import os
import tempfile
from typing import Dict
from agents.manager import DefaultJob
from agents.manager.progress_info import ProgressInfo
from agents.manager.training_job import TrainingJob
from agents.monitor.process_info import ProcessInfo


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
            assert isinstance(job_status, DefaultJob)
            assert job_status.config_filepath == config_path
            rel_path = os.path.splitext(os.path.relpath(config_path, start='./configs'))[0]
            expected_work_dir = os.path.join('./logs', rel_path)
            assert job_status.work_dir == expected_work_dir
            assert isinstance(job_status.progress, ProgressInfo)
            assert job_status.progress.completed_epochs == 5
            assert job_status.progress.early_stopped == False
            assert job_status.status == 'failed'  # No recent log updates, not running on GPU
            assert job_status.process_info is None  # Not running on GPU
            
        finally:
            os.chdir(original_cwd)
