from typing import List
from utils.automation.run_status import RunStatus


def get_progress(all_run_status: List[RunStatus]) -> float:
    """Get the progress of all config files in parallel."""
    all_run_progress = [run.progress for run in all_run_status]
    return round(sum(all_run_progress) / len(all_run_progress), 2)
