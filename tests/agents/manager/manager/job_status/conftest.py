"""
Local fixtures for agents.manager.job_status tests.

Copied or moved from tests/utils/automation/conftest.py to keep
manager tests self-contained.
"""
from typing import Optional, Dict, List
import os
import json
import pytest
import torch
from unittest.mock import Mock
from agents.monitor.gpu_status import GPUStatus
from agents.monitor.system_monitor import SystemMonitor
from agents.monitor.process_info import ProcessInfo
from utils.io.json import save_json


@pytest.fixture
def EXPECTED_FILES():
    """Standard expected files per epoch used by job_status tests."""
    return ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]


@pytest.fixture
def create_epoch_files():
    def _create_epoch_files(work_dir: str, epoch_idx: int, validation_score: Optional[float] = None) -> None:
        epoch_dir = os.path.join(work_dir, f"epoch_{epoch_idx}")
        os.makedirs(epoch_dir, exist_ok=True)

        # training_losses.pt
        training_losses_path = os.path.join(epoch_dir, "training_losses.pt")
        torch.save({"loss": torch.tensor(0.5)}, training_losses_path)

        # optimizer_buffer.json
        optimizer_buffer_path = os.path.join(epoch_dir, "optimizer_buffer.json")
        with open(optimizer_buffer_path, 'w') as f:
            json.dump({"lr": 0.001}, f)

        # validation_scores.json
        validation_scores_path = os.path.join(epoch_dir, "validation_scores.json")
        if validation_score is None:
            score_value = 0.4 - epoch_idx * 0.01
        else:
            score_value = validation_score
        validation_scores = {
            "aggregated": {"loss": score_value},
            "per_datapoint": {"loss": [score_value]}
        }
        with open(validation_scores_path, 'w') as f:
            json.dump(validation_scores, f)

    return _create_epoch_files


@pytest.fixture
def create_progress_json():
    def _create_progress_json(
        work_dir: str,
        completed_epochs: int,
        early_stopped: bool = False,
        early_stopped_at_epoch: Optional[int] = None,
        tot_epochs: int = 100
    ) -> None:
        progress_data = {
            "completed_epochs": completed_epochs,
            "progress_percentage": 100.0 if early_stopped else (completed_epochs / tot_epochs * 100.0),
            "early_stopped": early_stopped,
            "early_stopped_at_epoch": early_stopped_at_epoch
        }
        progress_file = os.path.join(work_dir, "progress.json")
        save_json(progress_data, progress_file)

    return _create_progress_json


@pytest.fixture
def create_real_config():
    def _create_real_config(
        config_path: str,
        work_dir: str,
        epochs: int = 100,
        early_stopping_enabled: bool = False,
        patience: int = 5
    ) -> None:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        config_content = f'''import torch
from metrics.wrappers import PyTorchMetricWrapper
from runners.early_stopping import EarlyStopping
from runners.trainers.supervised_single_task_trainer import SupervisedSingleTaskTrainer

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

    return _create_real_config


@pytest.fixture
def create_minimal_system_monitor_with_processes():
    def _create_minimal_system_monitor_with_processes(connected_gpus: List[GPUStatus]) -> Mock:
        """Return a SystemMonitor mock with connected_gpus as GPUStatus objects only.

        Enforces the latest API: callers must pass a list of GPUStatus.
        """
        assert isinstance(connected_gpus, list), f"connected_gpus must be list, got {type(connected_gpus)}"
        for i, gpu in enumerate(connected_gpus):
            if not isinstance(gpu, GPUStatus):
                raise TypeError(f"connected_gpus[{i}] must be GPUStatus, got {type(gpu)}")

        mock_monitor = Mock(spec=SystemMonitor)
        mock_monitor.server = 'test_server'
        mock_monitor.connected_gpus = connected_gpus
        return mock_monitor

    return _create_minimal_system_monitor_with_processes


@pytest.fixture
def setup_realistic_experiment_structure(create_real_config, create_epoch_files, create_minimal_system_monitor_with_processes):
    def _setup_realistic_experiment_structure(temp_root: str, experiments: List[tuple]) -> tuple:
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

            for epoch_idx in range(epochs_completed):
                create_epoch_files(work_dir, epoch_idx)

            if has_recent_logs:
                log_file = os.path.join(work_dir, "train_val_latest.log")
                with open(log_file, 'w') as f:
                    f.write("Recent training log")

            if target_status in ["running", "stuck"]:
                running_experiments.append(config_path)

        # imported at top: ProcessInfo, GPUStatus
        processes = [
            ProcessInfo(
                pid=f'1234{i}',
                user='testuser',
                cmd=f'python main.py --config-filepath {config_path}',
                start_time=f'Mon Jan  1 1{i}:00:00 2024'
            )
            for i, config_path in enumerate(running_experiments)
        ]
        connected_gpus_data = [
            GPUStatus(server='test_server', index=0, window_size=10, max_memory=0, processes=processes, connected=True)
        ] if running_experiments else [
            GPUStatus(server='test_server', index=0, window_size=10, max_memory=0, processes=[], connected=True)
        ]

        system_monitor = create_minimal_system_monitor_with_processes(connected_gpus_data)

        return config_files, work_dirs, {'test_server': system_monitor}

    return _setup_realistic_experiment_structure
