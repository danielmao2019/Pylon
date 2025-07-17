from typing import List, Optional, Literal, NamedTuple, Tuple, TypedDict, Dict
from functools import partial
import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor
import json
import torch
from utils.automation.cfg_log_conversion import get_work_dir, get_config
from utils.monitor.system_monitor import SystemMonitor
from utils.monitor.process_info import ProcessInfo
from utils.io.config import load_config
from utils.io.json import save_json
from utils.builders.builder import build_from_config


_RunStatus = Literal['running', 'finished', 'failed', 'stuck', 'outdated']


class ProgressInfo(TypedDict):
    completed_epochs: int
    progress_percentage: float
    early_stopped: bool
    early_stopped_at_epoch: Optional[int]


class RunStatus(NamedTuple):
    config: str
    work_dir: str
    progress: ProgressInfo
    status: _RunStatus
    process_info: Optional[ProcessInfo]


def get_run_status(
    config: str,
    expected_files: List[str],
    epochs: int,
    gpu_running_work_dirs: List[str],
    config_to_process_info: Dict[str, ProcessInfo],
    sleep_time: int = 86400,
    outdated_days: int = 30
) -> RunStatus:
    """Get the current status of a training run.

    Args:
        config: Path to the config file used for this run
        expected_files: List of files expected to be present in each epoch directory
        epochs: Total number of epochs for the training
        gpu_running_work_dirs: List of currently running jobs
        sleep_time: Time in seconds to consider a run as "stuck" if no updates
        outdated_days: Number of days after which a finished run is considered outdated

    Returns:
        RunStatus object containing the current status of the run
    """
    work_dir = get_work_dir(config)
    # Get enhanced progress info (ProgressInfo dict)
    progress = get_session_progress(work_dir, expected_files)
    
    # Get core metrics for status determination
    log_last_update = get_log_last_update(work_dir)
    epoch_last_update = get_epoch_last_update(work_dir, expected_files)

    # Determine if running based on log updates
    is_running_status = log_last_update is not None and (time.time() - log_last_update <= sleep_time)

    # Determine status based on metrics
    if is_running_status:
        status: _RunStatus = 'running'
    elif progress['completed_epochs'] >= epochs:  # Use dict access instead of int comparison
        # Check if finished run is outdated
        if epoch_last_update is not None and (time.time() - epoch_last_update > outdated_days * 24 * 60 * 60):
            status = 'outdated'
        else:
            status = 'finished'
    elif work_dir in gpu_running_work_dirs:
        status = 'stuck'
    else:
        status = 'failed'

    # Get ProcessInfo if this config is running on GPU
    process_info = config_to_process_info.get(config, None)

    return RunStatus(
        config=config,
        work_dir=work_dir,
        progress=progress,
        status=status,
        process_info=process_info
    )


def get_all_run_status(
    config_files: List[str],
    expected_files: List[str],
    epochs: int,
    sleep_time: int = 86400,
    outdated_days: int = 30,
    system_monitor: SystemMonitor = None,
) -> Dict[str, RunStatus]:
    """
    Args:
        config_files: List of config file paths
        expected_files: List of expected file patterns
        epochs: Total number of epochs
        sleep_time: Time to wait for the status to update
        outdated_days: Number of days to consider a run outdated
        system_monitor: System monitor (CPU + GPU)
    """
    assert isinstance(config_files, list)
    assert isinstance(expected_files, list)
    assert isinstance(epochs, int)
    assert isinstance(sleep_time, int)
    assert isinstance(outdated_days, int)
    assert isinstance(system_monitor, SystemMonitor)

    # Query all GPUs ONCE for efficiency
    all_connected_gpus = system_monitor.connected_gpus
    config_to_process_info = _build_config_to_process_mapping(all_connected_gpus)
    all_running_work_dirs = [get_work_dir(config) for config in config_to_process_info.keys()]

    with ThreadPoolExecutor() as executor:
        all_run_status = list(executor.map(
            partial(get_run_status,
                expected_files=expected_files,
                epochs=epochs,
                gpu_running_work_dirs=all_running_work_dirs,
                config_to_process_info=config_to_process_info,  # NEW parameter
                sleep_time=sleep_time,
                outdated_days=outdated_days,
            ), config_files
        ))

    # Convert list to mapping
    return {status.config: status for status in all_run_status}


def parse_config(cmd: str) -> str:
    """Extract config filepath from command string."""
    assert isinstance(cmd, str), f"{cmd=}"
    assert 'python' in cmd, f"{cmd=}"
    assert '--config-filepath' in cmd, f"{cmd=}"
    parts = cmd.split(' ')
    for idx, part in enumerate(parts):
        if part == "--config-filepath":
            return parts[idx+1]
    assert 0


def _build_config_to_process_mapping(connected_gpus: List) -> Dict[str, ProcessInfo]:
    """Build mapping from config file to ProcessInfo for running experiments."""
    config_to_process = {}
    for gpu in connected_gpus:
        for process in gpu['processes']:
            if 'python main.py --config-filepath' in process['cmd']:
                config = parse_config(process['cmd'])
                config_to_process[config] = process
    return config_to_process


def get_log_last_update(work_dir: str) -> Optional[float]:
    """Get the timestamp of the last log update.

    Returns:
        Timestamp of last update if logs exist, None otherwise
    """
    if not os.path.isdir(work_dir):
        return None
    logs = glob.glob(os.path.join(work_dir, "train_val*.log"))
    if len(logs) == 0:
        return None
    return max([os.path.getmtime(fp) for fp in logs])


def get_epoch_last_update(work_dir: str, expected_files: List[str]) -> Optional[float]:
    """Get the timestamp of the last epoch file update.

    Returns:
        Timestamp of last update if epoch files exist, None otherwise
    """
    if not os.path.isdir(work_dir):
        return None
    epoch_dirs = glob.glob(os.path.join(work_dir, "epoch_*"))
    if len(epoch_dirs) == 0:
        return None
    return max([
        os.path.getmtime(os.path.join(epoch_dir, filename))
        for epoch_dir in epoch_dirs
        for filename in expected_files
        if os.path.isfile(os.path.join(epoch_dir, filename))
    ])


def get_session_progress(work_dir: str, expected_files: List[str]) -> ProgressInfo:
    """Enhanced progress calculation with early stopping detection.
    
    Returns full progress info including early stopping details.
    
    Args:
        work_dir: Directory containing epoch results
        expected_files: List of expected files per epoch
        
    Returns:
        ProgressInfo dict with completed_epochs, progress_percentage, early_stopped, early_stopped_at_epoch
    """
    # Try fast path: read progress.json
    progress_file = os.path.join(work_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)  # Return full ProgressInfo dict
    
    # Slow path: re-compute and create progress.json
    return _compute_and_cache_progress(work_dir, expected_files)


def _compute_and_cache_progress(work_dir: str, expected_files: List[str]) -> ProgressInfo:
    """Compute progress and create/update progress.json file."""
    # Count completed epochs (original logic)
    completed_epochs = 0
    while True:
        if not check_epoch_finished(
            epoch_dir=os.path.join(work_dir, f"epoch_{completed_epochs}"),
            expected_files=expected_files,
            check_load=False,
        ):
            break
        completed_epochs += 1
    
    # Load config to detect early stopping
    config_path = get_config(work_dir)
    config = load_config(config_path)
    tot_epochs = config['epochs']
    early_stopping_config = config.get('early_stopping')
    
    early_stopped = False
    early_stopped_at_epoch = None
    
    if early_stopping_config and completed_epochs < tot_epochs:
        # Early stopping is configured and we have fewer epochs than expected
        # Detect if early stopping was triggered
        early_stopped, early_stopped_at_epoch = _detect_early_stopping(
            work_dir, expected_files, config, completed_epochs
        )
    
    # Create progress.json for future caching
    progress_data = {
        "completed_epochs": completed_epochs,
        "progress_percentage": 100.0 if early_stopped else (completed_epochs / tot_epochs * 100.0),
        "early_stopped": early_stopped,
        "early_stopped_at_epoch": early_stopped_at_epoch
    }
    
    progress_file = os.path.join(work_dir, "progress.json")
    save_json(progress_data, progress_file)
    
    # Return the full progress info dict
    return progress_data


def _detect_early_stopping(work_dir: str, expected_files: List[str], config: dict, completed_epochs: int) -> Tuple[bool, Optional[int]]:
    """Detect if early stopping was triggered."""
    # Build metric and early stopping objects to detect early stopping
    metric = build_from_config(config['metric']) if config.get('metric') else None
    
    if metric is None:
        return False, None
        
    early_stopping_config = config['early_stopping']
    
    early_stopping = build_from_config(
        config=early_stopping_config,
        work_dir=work_dir,
        tot_epochs=config['epochs'],
        metric=metric,
        expected_files=expected_files,
        logger=None
    )
    
    # Check if early stopping would be triggered at the last completed epoch
    if completed_epochs > early_stopping.patience:
        last_epoch = completed_epochs - 1
        if early_stopping.was_triggered_at_epoch(last_epoch):
            return True, last_epoch + 1
    
    return False, None


def check_file_loadable(filepath: str) -> bool:
    """Check if a file can be loaded (json or pt)."""
    assert os.path.isfile(filepath)
    assert filepath.endswith(".json") or filepath.endswith(".pt")
    result: bool = True
    if filepath.endswith(".json"):
        try:
            with open(filepath, mode='r') as f:
                _ = json.load(f)
        except:
            result = False
    elif filepath.endswith(".pt"):
        try:
            _ = torch.load(filepath)
        except:
            result = False
    else:
        assert 0
    return result


def check_epoch_finished(epoch_dir: str, expected_files: List[str], check_load: Optional[bool] = True) -> bool:
    r"""Check if an epoch is finished based on three criteria:
        1. File exists.
        2. File non-empty.
        3. File load-able (if check_load is True).
    """
    if not os.path.isdir(epoch_dir):
        return False
    return all([
        os.path.isfile(os.path.join(epoch_dir, filename)) and
        os.path.getsize(os.path.join(epoch_dir, filename)) > 0 and
        ((not check_load) or check_file_loadable(os.path.join(epoch_dir, filename)))
        for filename in expected_files
    ])
