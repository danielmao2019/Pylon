from typing import List
from utils.automation.run_status import get_all_run_status


def get_progress(
    config_files: List[str],
    expected_files: List[str],
    epochs: int,
    sleep_time: int,
    outdated_days: int,
    servers: List[str],
) -> float:
    all_run_status = get_all_run_status(config_files, expected_files, epochs, sleep_time, outdated_days, servers)
    all_run_progress = [run.progress for run in all_run_status]
    return round(sum(all_run_progress) / len(all_run_progress), 2)
