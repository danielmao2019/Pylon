"""
Manager module fixtures shared across tests. No imports from this file; use fixtures via parameters only.
"""
import os
import json
import tempfile
import time
import pytest
from typing import Optional
import torch
from utils.io.json import save_json


def make_command(config_path: str) -> str:
    """Helper to build the canonical launch command for a config file."""
    return f"python main.py --config-filepath {config_path}"


@pytest.fixture
def temp_manager_root():
    """Create a temporary root and chdir so ./configs and ./logs resolve correctly.

    Yields (root_path) and restores CWD after the test.
    """
    with tempfile.TemporaryDirectory() as root:
        configs = os.path.join(root, 'configs')
        logs = os.path.join(root, 'logs')
        os.makedirs(configs, exist_ok=True)
        os.makedirs(logs, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(root)
        try:
            yield root
        finally:
            os.chdir(cwd)


@pytest.fixture
def write_config():
    """Fixture to write a config file under ./configs with provided dict as 'config'.

    Usage: write_config('exp.py', {'epochs': 100}) -> returns './configs/exp.py'
    """
    def _write(name: str, config: dict) -> str:
        path = os.path.join('./configs', name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write('config = ' + json.dumps(config) + '\n')
        return path
    return _write


@pytest.fixture
def make_trainer_epoch():
    """Create required trainer epoch files in ./logs/<exp>/epoch_<idx>.

    Valid JSON files for optimizer_buffer.json and validation_scores.json.
    training_losses.pt is created as non-empty bytes (content not loaded in tests using fast path).
    Returns the epoch directory path.
    """
    def _make(exp_name: str, idx: int = 0) -> str:
        work = os.path.join('./logs', exp_name)
        epoch_dir = os.path.join(work, f'epoch_{idx}')
        os.makedirs(epoch_dir, exist_ok=True)
        with open(os.path.join(epoch_dir, 'validation_scores.json'), 'w') as f:
            json.dump({'acc': 0.9}, f)
        with open(os.path.join(epoch_dir, 'optimizer_buffer.json'), 'w') as f:
            json.dump({'lr': 1e-3}, f)
        with open(os.path.join(epoch_dir, 'training_losses.pt'), 'wb') as f:
            f.write(b'\x00')
        return epoch_dir
    return _make


@pytest.fixture
def write_eval_scores():
    """Create evaluation_scores.json under ./logs/<exp> with valid JSON.

    Returns the work directory path.
    """
    def _write(exp_name: str) -> str:
        work = os.path.join('./logs', exp_name)
        os.makedirs(work, exist_ok=True)
        with open(os.path.join(work, 'evaluation_scores.json'), 'w') as f:
            json.dump({'aggregated': {'acc': 1.0}, 'per_datapoint': {'acc': [1.0]}}, f)
        return work
    return _write


@pytest.fixture
def touch_log():
    """Touch a log file in ./logs/<exp> matching TrainingJob.get_log_pattern().

    touch_log('exp', age_seconds=0) creates a fresh log; set age_seconds to age it.
    Returns the log path.
    """
    def _touch(exp_name: str, *, age_seconds: int = 0, name: str = 'train_val.log') -> str:
        work = os.path.join('./logs', exp_name)
        os.makedirs(work, exist_ok=True)
        log_path = os.path.join(work, name)
        with open(log_path, 'a'):
            os.utime(log_path, None)
        if age_seconds > 0:
            t = time.time() - age_seconds
            os.utime(log_path, (t, t))
        return log_path
    return _touch

@pytest.fixture
def create_epoch_files():
    def _create_epoch_files(work_dir: str, epoch_idx: int, validation_score: Optional[float] = None) -> None:
        epoch_dir = os.path.join(work_dir, f"epoch_{epoch_idx}")
        os.makedirs(epoch_dir, exist_ok=True)

        torch.save({"loss": torch.tensor(0.5)}, os.path.join(epoch_dir, "training_losses.pt"))
        with open(os.path.join(epoch_dir, "optimizer_buffer.json"), 'w') as f:
            json.dump({"lr": 0.001}, f)

        score_value = 0.4 - epoch_idx * 0.01 if validation_score is None else validation_score
        validation_scores = {"aggregated": {"loss": score_value}, "per_datapoint": {"loss": [score_value]}}
        with open(os.path.join(epoch_dir, "validation_scores.json"), 'w') as f:
            json.dump(validation_scores, f)
    return _create_epoch_files


@pytest.fixture
def create_progress_json():
    def _create_progress_json(
        work_dir: str,
        completed_epochs: int,
        early_stopped: bool = False,
        early_stopped_at_epoch: Optional[int] = None,
        tot_epochs: int = 100,
    ) -> None:
        progress_data = {
            "completed_epochs": completed_epochs,
            "progress_percentage": 100.0 if early_stopped else (completed_epochs / tot_epochs * 100.0),
            "early_stopped": early_stopped,
            "early_stopped_at_epoch": early_stopped_at_epoch,
        }
        save_json(progress_data, os.path.join(work_dir, "progress.json"))
    return _create_progress_json


@pytest.fixture
def create_real_config():
    def _create_real_config(
        config_path: str,
        work_dir: str,
        epochs: int = 100,
        early_stopping_enabled: bool = False,
        patience: int = 5,
    ) -> None:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        content = f'''import torch
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
            content += f"""
    'early_stopping': {{
        'class': EarlyStopping,
        'args': {{
            'enabled': True,
            'epochs': {patience},
        }},
    }},"""
        content += """
}
"""
        with open(config_path, 'w') as f:
            f.write(content)
    return _create_real_config


@pytest.fixture
def EXPECTED_FILES():
    return ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]


@pytest.fixture
def create_system_monitor_with_processes():
    """Return a factory that creates a monitors dict with a SystemMonitor-like object exposing connected_gpus.

    Usage:
        monitors = create_system_monitor_with_processes([
            "python main.py --config-filepath ./configs/exp.py",
        ])
        manager = Manager(commands=[...], system_monitors=monitors, ...)
    """
    from agents.monitor.system_monitor import SystemMonitor
    from agents.monitor.gpu_status import GPUStatus
    from agents.monitor.process_info import ProcessInfo

    class LocalSystemMonitor(SystemMonitor):
        def __init__(self, server: str, window_size: int, gpus):
            super().__init__(server=server, window_size=window_size)
            self._local_connected_gpus = gpus
        @property
        def connected_gpus(self):
            return self._local_connected_gpus

    def _factory(cmds):
        processes = [ProcessInfo(pid=str(i+1), user='u', cmd=cmd, start_time='t') for i, cmd in enumerate(cmds)]
        gpus = [GPUStatus(server='localhost', index=0, window_size=10, max_memory=0, processes=processes, connected=True)]
        sm = LocalSystemMonitor(server='localhost', window_size=10, gpus=gpus)
        return {'localhost': sm}

    return _factory
