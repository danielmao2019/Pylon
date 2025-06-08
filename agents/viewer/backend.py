from typing import List, Dict, Any
import os
import json
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.automation.cfg_log_conversion import get_work_dir
from utils.automation.run_status import get_session_progress


def get_progress(config_files: List[str], expected_files: List[str], epochs: int) -> float:
    """Get the progress of all config files in parallel."""
    def process_config(config_file: str) -> int:
        work_dir = get_work_dir(config_file)
        cur_epochs = get_session_progress(work_dir=work_dir, expected_files=expected_files, epochs=epochs)
        return int(cur_epochs / epochs * 100)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_config, config_files))
    
    # Calculate average progress
    return round(sum(results) / len(results), 2)
