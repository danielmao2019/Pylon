from typing import List, Optional, Literal, NamedTuple, Dict
from functools import partial
import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor
from utils.automation.cfg_log_conversion import get_work_dir
from utils.monitor.system_monitor import SystemMonitor
from utils.monitor.process_info import ProcessInfo
from utils.automation.run_status.session_progress import get_session_progress, ProgressInfo


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

_RunStatus = Literal['running', 'finished', 'failed', 'stuck', 'outdated']


class RunStatus(NamedTuple):
    config: str
    work_dir: str
    progress: ProgressInfo
    status: _RunStatus
    process_info: Optional[ProcessInfo]


# ============================================================================
# MAIN PUBLIC FUNCTIONS
# ============================================================================

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

    with ThreadPoolExecutor() as executor:
        all_run_status = list(executor.map(
            partial(get_run_status,
                expected_files=expected_files,
                epochs=epochs,
                config_to_process_info=config_to_process_info,
                sleep_time=sleep_time,
                outdated_days=outdated_days,
            ), config_files
        ))

    # Convert list to mapping
    return {status.config: status for status in all_run_status}


def get_run_status(
    config: str,
    expected_files: List[str],
    epochs: int,
    config_to_process_info: Dict[str, ProcessInfo],
    sleep_time: int = 86400,
    outdated_days: int = 30
) -> RunStatus:
    """Get the current status of a training run.

    Args:
        config: Path to the config file used for this run
        expected_files: List of files expected to be present in each epoch directory
        epochs: Total number of epochs for the training
        config_to_process_info: Mapping from config to ProcessInfo for running experiments
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
    elif config in config_to_process_info:
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


# ============================================================================
# HELPER FUNCTIONS FOR STATUS DETERMINATION
# ============================================================================

def _build_config_to_process_mapping(connected_gpus: List) -> Dict[str, ProcessInfo]:
    """Build mapping from config file to ProcessInfo for running experiments."""
    config_to_process = {}
    for gpu in connected_gpus:
        for process in gpu['processes']:
            if 'python main.py --config-filepath' in process['cmd']:
                config = parse_config(process['cmd'])
                config_to_process[config] = process
    return config_to_process


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