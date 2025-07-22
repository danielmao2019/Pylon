from typing import List
from utils.automation.run_status import get_all_run_status
from utils.monitor.system_monitor import SystemMonitor


def get_progress(
    config_files: List[str],
    epochs: int,
    sleep_time: int,
    outdated_days: int,
    system_monitor: SystemMonitor,
) -> float:
    """
    Args:
        config_files: List of config file paths
        epochs: Total number of epochs
        sleep_time: Time to wait for the status to update
        outdated_days: Number of days to consider a run outdated
        system_monitor: System monitor (CPU + GPU)
    """
    assert isinstance(config_files, list)
    assert isinstance(epochs, int)
    assert isinstance(sleep_time, int)
    assert isinstance(outdated_days, int)
    assert isinstance(system_monitor, SystemMonitor)

    all_run_status = get_all_run_status(config_files, epochs, sleep_time, outdated_days, system_monitor)
    all_run_progress = [run.progress.progress_percentage for run in all_run_status.values()]
    
    # Handle empty config_files case
    if len(all_run_progress) == 0:
        return 0.0
    
    return round(sum(all_run_progress) / len(all_run_progress), 2)
