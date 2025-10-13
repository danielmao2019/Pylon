from typing import List
from agents.manager import Manager
from agents.monitor.system_monitor import SystemMonitor


def get_progress(
    commands: List[str],
    epochs: int,
    sleep_time: int,
    outdated_days: int,
    system_monitor: SystemMonitor,
    force_progress_recompute: bool = False,
) -> float:
    """
    Args:
        commands: List of runtime command strings
        epochs: Total number of epochs
        sleep_time: Time to wait for the status to update
        outdated_days: Number of days to consider a run outdated
        system_monitor: System monitor (CPU + GPU)
        force_progress_recompute: If True, bypass cache and recompute progress from scratch
    """
    assert isinstance(commands, list), f"commands must be list, got {type(commands)}"
    assert len(commands) > 0, f"commands must not be empty, got {len(commands)} entries"
    assert isinstance(epochs, int), f"epochs must be int, got {type(epochs)}"
    assert isinstance(sleep_time, int), f"sleep_time must be int, got {type(sleep_time)}"
    assert isinstance(outdated_days, int), f"outdated_days must be int, got {type(outdated_days)}"
    assert isinstance(system_monitor, SystemMonitor), f"system_monitor must be SystemMonitor, got {type(system_monitor)}"
    assert isinstance(force_progress_recompute, bool), f"force_progress_recompute must be bool, got {type(force_progress_recompute)}"

    monitor_map = {system_monitor.server: system_monitor}
    manager = Manager(
        commands=commands,
        epochs=epochs,
        system_monitors=monitor_map,
        sleep_time=sleep_time,
        outdated_days=outdated_days,
        force_progress_recompute=force_progress_recompute,
    )
    all_job_status = manager.build_jobs()
    all_job_progress = [s.progress.progress_percentage for s in all_job_status.values()]

    return round(sum(all_job_progress) / len(all_job_progress), 2)
