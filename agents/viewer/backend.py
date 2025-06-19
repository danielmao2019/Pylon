from typing import List
from utils.automation.run_status import get_all_run_status
from utils.monitor.gpu_monitor import GPUMonitor


def get_progress(
    config_files: List[str],
    expected_files: List[str],
    epochs: int,
    sleep_time: int,
    outdated_days: int,
    gpu_monitor: GPUMonitor,
) -> float:
    """
    Args:
        config_files: List of config file paths
        expected_files: List of expected file patterns
        epochs: Total number of epochs
        sleep_time: Time to wait for the status to update
        outdated_days: Number of days to consider a run outdated
        gpu_monitor: GPU monitor
    """
    assert isinstance(config_files, list)
    assert isinstance(expected_files, list)
    assert isinstance(epochs, int)
    assert isinstance(sleep_time, int)
    assert isinstance(outdated_days, int)
    assert isinstance(gpu_monitor, GPUMonitor)

    all_run_status = get_all_run_status(config_files, expected_files, epochs, sleep_time, outdated_days, gpu_monitor)
    all_run_progress = [run.progress for run in all_run_status]
    return round(sum(all_run_progress) / len(all_run_progress), 2)
