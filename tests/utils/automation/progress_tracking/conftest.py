"""
Shared test fixtures and helper functions for run_status tests.

This conftest.py provides common test data creation functions that are
used across multiple test files to eliminate duplication.
"""
from typing import Optional, Dict, List
import os
import tempfile
import json
import pytest
import torch
from unittest.mock import Mock
from utils.automation.progress_tracking import ProgressInfo
from utils.monitor.system_monitor import SystemMonitor
from utils.io.json import save_json


# ============================================================================
# COMMON TEST HELPER CLASSES AS FIXTURES
# ============================================================================

@pytest.fixture
def ConcreteProgressTracker():
    """Fixture that provides ConcreteProgressTracker class for testing BaseProgressTracker."""
    from typing import List, Literal
    from utils.automation.progress_tracking.base_progress_tracker import BaseProgressTracker, ProgressInfo
    
    class ConcreteProgressTrackerImpl(BaseProgressTracker):
        """Concrete implementation for testing BaseProgressTracker functionality."""
        
        def __init__(self, work_dir: str, config=None, runner_type: Literal['trainer', 'evaluator'] = 'trainer'):
            super().__init__(work_dir, config)
            self._runner_type = runner_type
            self._test_progress_result = None
        
        def get_runner_type(self) -> Literal['trainer', 'evaluator']:
            return self._runner_type
        
        def get_expected_files(self) -> List[str]:
            return ["test_file.json", "another_file.pt"]
        
        def get_log_pattern(self) -> str:
            return "test_*.log"
        
        def calculate_progress(self) -> ProgressInfo:
            """Mock implementation for testing."""
            if self._test_progress_result is None:
                return ProgressInfo(
                    completed_epochs=10,
                    progress_percentage=50.0,
                    early_stopped=False,
                    early_stopped_at_epoch=None,
                    runner_type=self._runner_type,
                    total_epochs=20
                )
            return self._test_progress_result
        
        def set_test_progress_result(self, result: ProgressInfo):
            """Test helper to control calculate_progress output."""
            self._test_progress_result = result
    
    return ConcreteProgressTrackerImpl


# ============================================================================
# COMMON TEST DATA CREATION HELPERS
# ============================================================================

def create_epoch_files(work_dir: str, epoch_idx: int, validation_score: Optional[float] = None) -> None:
    """Create all expected files for a completed epoch.
    
    This is the most comprehensive version that supports optional validation scores.
    Used across all test files for creating realistic epoch data.
    """
    epoch_dir = os.path.join(work_dir, f"epoch_{epoch_idx}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Create training_losses.pt
    training_losses_path = os.path.join(epoch_dir, "training_losses.pt")
    torch.save({"loss": torch.tensor(0.5)}, training_losses_path)
    
    # Create optimizer_buffer.json
    optimizer_buffer_path = os.path.join(epoch_dir, "optimizer_buffer.json")
    with open(optimizer_buffer_path, 'w') as f:
        json.dump({"lr": 0.001}, f)
    
    # Create validation_scores.json with realistic structure
    validation_scores_path = os.path.join(epoch_dir, "validation_scores.json")
    
    if validation_score is None:
        score_value = 0.4 - epoch_idx * 0.01  # Improving scores (decreasing loss)
    else:
        score_value = validation_score
        
    validation_scores = {
        "aggregated": {"loss": score_value},
        "per_datapoint": {"loss": [score_value]}
    }
    with open(validation_scores_path, 'w') as f:
        json.dump(validation_scores, f)


def create_progress_json(
    work_dir: str, 
    completed_epochs: int, 
    early_stopped: bool = False, 
    early_stopped_at_epoch: Optional[int] = None, 
    tot_epochs: int = 100
) -> None:
    """Create progress.json file for fast path testing."""
    progress_data: ProgressInfo = {
        "completed_epochs": completed_epochs,
        "progress_percentage": 100.0 if early_stopped else (completed_epochs / tot_epochs * 100.0),
        "early_stopped": early_stopped,
        "early_stopped_at_epoch": early_stopped_at_epoch
    }
    progress_file = os.path.join(work_dir, "progress.json")
    save_json(progress_data, progress_file)


def create_real_config(
    config_path: str, 
    work_dir: str, 
    epochs: int = 100, 
    early_stopping_enabled: bool = False, 
    patience: int = 5
) -> None:
    """Create a real config file for testing.
    
    This is the most comprehensive version that supports early stopping configuration.
    Used across all test files for creating realistic config files.
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    config_content = f'''import torch
from metrics.wrappers import PyTorchMetricWrapper
from runners.early_stopping import EarlyStopping
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer

config = {{
    'runner': SupervisedSingleTaskTrainer,
    'epochs': {epochs},
    'work_dir': '{work_dir}',
    'metric': {{
        'class': PyTorchMetricWrapper,
        'args': {{
            'metric': torch.nn.MSELoss(reduction='mean'),
        }},
    }},'''
    
    if early_stopping_enabled:
        config_content += f'''
    'early_stopping': {{
        'class': EarlyStopping,
        'args': {{
            'enabled': True,
            'epochs': {patience},
        }},
    }},'''
    
    config_content += '''
}
'''
    
    with open(config_path, 'w') as f:
        f.write(config_content)


def create_minimal_system_monitor_with_processes(connected_gpus_data: List[Dict]) -> Mock:
    """Create a minimal SystemMonitor mock with realistic data.
    
    This is the ONLY necessary mock in our test suite because:
    1. SystemMonitor has complex internal state and SSH dependencies
    2. We only need the connected_gpus property for our testing
    3. All other data structures are real (ProcessInfo, RunStatus, etc.)
    
    This mock only mocks the SystemMonitor interface, not the data.
    """
    mock_monitor = Mock(spec=SystemMonitor)
    mock_monitor.connected_gpus = connected_gpus_data
    return mock_monitor


def setup_realistic_experiment_structure(temp_root: str, experiments: List[tuple]) -> tuple:
    """Set up a realistic experiment directory structure.
    
    Args:
        temp_root: Root temporary directory
        experiments: List of (exp_name, target_status, epochs_completed, has_recent_logs) tuples
        
    Returns:
        (config_files, work_dirs, system_monitor) tuple
    """
    logs_dir = os.path.join(temp_root, "logs")
    configs_dir = os.path.join(temp_root, "configs")
    
    config_files = []
    work_dirs = {}
    running_experiments = []
    
    for exp_name, target_status, epochs_completed, has_recent_logs in experiments:
        work_dir = os.path.join(logs_dir, exp_name)
        config_path = os.path.join(configs_dir, f"{exp_name}.py")
        
        os.makedirs(work_dir, exist_ok=True)
        create_real_config(config_path, work_dir, epochs=100)
        config_files.append(config_path)
        work_dirs[config_path] = work_dir
        
        # Create epoch files
        for epoch_idx in range(epochs_completed):
            create_epoch_files(work_dir, epoch_idx)
        
        # Create recent logs if needed
        if has_recent_logs:
            log_file = os.path.join(work_dir, "train_val_latest.log")
            with open(log_file, 'w') as f:
                f.write("Recent training log")
        
        # Track which experiments should be running on GPU
        if target_status in ["running", "stuck"]:
            running_experiments.append(config_path)
    
    # Create realistic GPU process data
    from utils.monitor.process_info import ProcessInfo
    connected_gpus_data = [
        {
            'server': 'test_server',
            'index': 0,
            'processes': [
                ProcessInfo(
                    pid=f'1234{i}',
                    user='testuser',
                    cmd=f'python main.py --config-filepath {config_path}',
                    start_time=f'Mon Jan  1 1{i}:00:00 2024'
                )
                for i, config_path in enumerate(running_experiments)
            ]
        }
    ] if running_experiments else [{'server': 'test_server', 'index': 0, 'processes': []}]
    
    # Create minimal SystemMonitor mock with realistic data
    system_monitor = create_minimal_system_monitor_with_processes(connected_gpus_data)
    
    return config_files, work_dirs, system_monitor


# ============================================================================
# COMMON TEST CONSTANTS
# ============================================================================

# Standard expected files for most tests
EXPECTED_FILES = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]


# ============================================================================
# PYTEST FIXTURES (Optional - for commonly used test setup)
# ============================================================================

@pytest.fixture
def temp_experiment_setup():
    """Fixture that provides a temporary directory with basic experiment structure."""
    with tempfile.TemporaryDirectory() as temp_root:
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)
        
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            yield {
                'temp_root': temp_root,
                'logs_dir': logs_dir,
                'configs_dir': configs_dir
            }
        finally:
            os.chdir(original_cwd)
