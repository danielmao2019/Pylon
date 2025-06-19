from typing import List, Optional, Literal, NamedTuple
from functools import partial
import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor
import json
import torch
from utils.automation.cfg_log_conversion import get_work_dir
from utils.monitor.gpu_monitor import GPUMonitor


_RunStatus = Literal['running', 'finished', 'failed', 'stuck', 'outdated']


class RunStatus(NamedTuple):
    config: str
    work_dir: str
    progress: int
    status: _RunStatus


def get_run_status(
    config: str,
    expected_files: List[str],
    epochs: int,
    gpu_running_work_dirs: List[str],
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
    # Get core metrics
    log_last_update = get_log_last_update(work_dir)
    epoch_last_update = get_epoch_last_update(work_dir, expected_files)
    progress = get_session_progress(work_dir, expected_files)

    # Determine if running based on log updates
    is_running_status = log_last_update is not None and (time.time() - log_last_update <= sleep_time)

    # Determine status based on metrics
    if is_running_status:
        status: _RunStatus = 'running'
    elif progress == epochs:
        # Check if finished run is outdated
        if epoch_last_update is not None and (time.time() - epoch_last_update > outdated_days * 24 * 60 * 60):
            status = 'outdated'
        else:
            status = 'finished'
    elif work_dir in gpu_running_work_dirs:
        status = 'stuck'
    else:
        status = 'failed'

    return RunStatus(
        config=config,
        work_dir=work_dir,
        progress=progress,
        status=status
    )


def get_all_run_status(
    config_files: List[str],
    expected_files: List[str],
    epochs: int,
    sleep_time: int = 86400,
    outdated_days: int = 30,
    gpu_monitor: GPUMonitor = None,
) -> List[RunStatus]:
    """
    Args:
        config_files: List of config file paths
        expected_files: List of expected file patterns
        epochs: Total number of epochs
        sleep_time: Time to wait for the status to update
        outdated_days: Number of days to consider a run outdated
        servers: List of servers
    """
    assert isinstance(config_files, list)
    assert isinstance(expected_files, list)
    assert isinstance(epochs, int)
    assert isinstance(sleep_time, int)
    assert isinstance(outdated_days, int)
    assert isinstance(gpu_monitor, GPUMonitor)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(find_running, servers))
    all_running = [run for server_runs in results for run in server_runs]
    all_running_work_dirs = list(map(lambda x: get_work_dir(parse_config(x['command'])), all_running))

    with ThreadPoolExecutor() as executor:
        all_run_status = list(executor.map(
            partial(get_run_status,
                expected_files=expected_files,
                epochs=epochs,
                gpu_running_work_dirs=all_running_work_dirs,
                sleep_time=sleep_time,
                outdated_days=outdated_days,
            ), config_files
        ))

    return all_run_status


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


def get_session_progress(work_dir: str, expected_files: List[str]) -> int:
    """Get the current progress (number of completed epochs).

    Returns:
        Number of completed epochs
    """
    idx = 0
    while True:
        if not check_epoch_finished(
            epoch_dir=os.path.join(work_dir, f"epoch_{idx}"),
            expected_files=expected_files,
            check_load=False,
        ):
            break
        idx += 1
    return idx


def parse_config(cmd: str) -> str:
    """Extract config filepath from command string."""
    assert 'python' in cmd, f"{cmd=}"
    assert '--config-filepath' in cmd, f"{cmd=}"
    parts = cmd.split(' ')
    for idx, part in enumerate(parts):
        if part == "--config-filepath":
            return parts[idx+1]
    assert 0


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
