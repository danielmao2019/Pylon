import os
import tempfile

from agents.manager.default_job import DefaultJob
from agents.manager.manager import Manager
from agents.manager.progress_info import ProgressInfo



def test_get_all_job_status_returns_mapping(create_real_config, create_epoch_files, create_system_monitor_with_processes):
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
        
        # Create minimal SystemMonitor mapping (no running processes)
        monitor_map = create_system_monitor_with_processes([])
        
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            commands = [f"python main.py --config-filepath {p}" for p in config_files]
            manager = Manager(commands=commands, epochs=100, system_monitors=monitor_map)
            result = manager.build_jobs()
            
            # Should return mapping instead of list
            assert isinstance(result, dict), f"Expected dict, got {type(result)}"
            assert len(result) == 3
            
            # Check that keys are config paths and values are BaseJob objects
            for config_path in config_files:
                key = f"python main.py --config-filepath {config_path}"
                assert key in result
                assert isinstance(result[key], DefaultJob)
                assert result[key].config_filepath == config_path
                assert isinstance(result[key].progress, ProgressInfo)
                assert hasattr(result[key].progress, 'completed_epochs')
                
        finally:
            os.chdir(original_cwd)
