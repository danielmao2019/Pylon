from typing import List, Optional, Literal, Dict
from functools import partial
import os
import glob
import time
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from utils.automation.cfg_log_conversion import get_work_dir
from utils.monitor.system_monitor import SystemMonitor
from utils.monitor.gpu_status import GPUStatus
from utils.monitor.process_info import ProcessInfo
from agents.tracker import ProgressInfo, create_tracker
from utils.io.config import load_config


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

_JobStatus = Literal['running', 'finished', 'failed', 'stuck', 'outdated']


@dataclass
class JobStatus:
    """Status information for a training job."""
    config: str
    work_dir: str
    progress: ProgressInfo
    status: _JobStatus
    process_info: Optional[ProcessInfo] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# ============================================================================
# MAIN PUBLIC FUNCTIONS
# ============================================================================

def get_all_job_status(
    config_files: List[str],
    epochs: int,
    sleep_time: int = 86400,
    outdated_days: int = 30,
    system_monitor: SystemMonitor = None,
    force_progress_recompute: bool = False,
) -> Dict[str, JobStatus]:
    """
    Args:
        config_files: List of config file paths
        epochs: Total number of epochs
        sleep_time: Time to wait for the status to update
        outdated_days: Number of days to consider a run outdated
        system_monitor: System monitor (CPU + GPU)
        force_progress_recompute: If True, bypass cache and recompute progress from scratch
    """
    assert isinstance(config_files, list)
    assert isinstance(epochs, int)
    assert isinstance(sleep_time, int)
    assert isinstance(outdated_days, int)
    assert isinstance(system_monitor, SystemMonitor)

    # Query all GPUs ONCE for efficiency
    all_connected_gpus = system_monitor.connected_gpus
    config_to_process_info = _build_config_to_process_mapping(all_connected_gpus)

    with ThreadPoolExecutor() as executor:
        all_job_status = {
            config: status 
            for config, status in zip(
                config_files,
                executor.map(
                    partial(get_job_status,
                        epochs=epochs,
                        config_to_process_info=config_to_process_info,
                        sleep_time=sleep_time,
                        outdated_days=outdated_days,
                        force_progress_recompute=force_progress_recompute,
                    ), config_files
                )
            )
        }
    
    # Validate that status.config matches the keys (config_files)
    assert set(all_job_status.keys()) == {status.config for status in all_job_status.values()}, \
        f"Mismatch between config_files and status.config values"
    
    return all_job_status


def get_job_status(
    config: str,
    epochs: int,
    config_to_process_info: Dict[str, ProcessInfo],
    sleep_time: int = 86400,
    outdated_days: int = 30,
    force_progress_recompute: bool = False
) -> JobStatus:
    """Get the current status of a training job.

    Args:
        config: Path to the config file used for this job
        epochs: Total number of epochs for the training
        config_to_process_info: Mapping from config to ProcessInfo for running experiments
        sleep_time: Time in seconds to consider a job as "stuck" if no updates
        outdated_days: Number of days after which a finished job is considered outdated
        force_progress_recompute: If True, bypass cache and recompute progress from scratch

    Returns:
        JobStatus object containing the current status of the job
    """
    work_dir = get_work_dir(config)
    config_dict = load_config(config)  # Load actual config
    
    # Create progress tracker (handles both trainer and evaluator)
    tracker = create_tracker(work_dir, config_dict)
    progress = tracker.get_progress(force_progress_recompute=force_progress_recompute)
    
    # Determine status using existing logic but with tracker data
    log_last_update = get_log_last_update(work_dir, tracker.get_log_pattern())
    epoch_last_update = get_epoch_last_update(work_dir, tracker.get_expected_files())
    
    is_running_status = log_last_update is not None and (time.time() - log_last_update <= sleep_time)
    
    # Determine completion based on epochs parameter (not tracker)
    is_complete = progress.completed_epochs >= epochs
    
    if is_running_status:
        status: _JobStatus = 'running'
    elif is_complete:
        if epoch_last_update is not None and (time.time() - epoch_last_update > outdated_days * 24 * 60 * 60):
            status = 'outdated'
        else:
            status = 'finished'
    elif config in config_to_process_info:
        status = 'stuck'
    else:
        status = 'failed'
    
    process_info = config_to_process_info.get(config, None)
    
    return JobStatus(
        config=config,
        work_dir=work_dir,
        progress=progress,  # Now includes runner_type and enhanced info
        status=status,
        process_info=process_info
    )


# ============================================================================
# HELPER FUNCTIONS FOR STATUS DETERMINATION
# ============================================================================

def _build_config_to_process_mapping(connected_gpus: List[GPUStatus]) -> Dict[str, ProcessInfo]:
    """Build mapping from config file to ProcessInfo for running experiments."""
    config_to_process = {}
    for gpu in connected_gpus:
        for process in gpu.processes:
            # Assert that process is already a ProcessInfo instance
            assert isinstance(process, ProcessInfo), f"Expected ProcessInfo instance, got {type(process)}"
            
            if 'python main.py --config-filepath' in process.cmd:
                config = parse_config(process.cmd)
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


def get_log_last_update(work_dir: str, log_pattern: str = "train_val*.log") -> Optional[float]:
    """Get the timestamp of the last log update.

    Args:
        work_dir: Directory to check for log files
        log_pattern: Glob pattern for log files (e.g., "train_val*.log" or "eval_*.log")

    Returns:
        Timestamp of last update if logs exist, None otherwise
    """
    if not os.path.isdir(work_dir):
        return None
    logs = glob.glob(os.path.join(work_dir, log_pattern))
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
