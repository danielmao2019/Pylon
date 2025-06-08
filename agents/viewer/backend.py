from typing import List
from utils.automation.cfg_log_conversion import get_work_dir
from utils.automation.run_status import get_session_progress


def get_progress(config_files: List[str], expected_files: List[str], epochs: int) -> float:
    """Calculate the overall progress across all config files.
    
    Args:
        config_files: List of config file paths
        expected_files: List of expected file patterns
        epochs: Total number of epochs
        
    Returns:
        float: Average progress percentage across all config files
    """
    result: int = 0
    for config_file in config_files:
        work_dir = get_work_dir(config_file)
        cur_epochs = get_session_progress(work_dir=work_dir, expected_files=expected_files, epochs=epochs)
        percentage = int(cur_epochs / epochs * 100)
        result += percentage
    result: float = round(result / len(config_files), 2)
    return result
