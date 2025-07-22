from typing import List, Optional, Tuple
import os
import json
import torch
from dataclasses import dataclass, asdict
from utils.automation.cfg_log_conversion import get_config
from utils.io.config import load_config
from utils.io.json import save_json
from utils.builders.builder import build_from_config

# Import the unified ProgressInfo from base module
from .base_progress_tracker import ProgressInfo


# ============================================================================
# MAIN PUBLIC FUNCTIONS
# ============================================================================

def get_session_progress(work_dir: str, expected_files: List[str]) -> ProgressInfo:
    """Enhanced progress calculation with early stopping detection.
    
    Returns full progress info including early stopping details.
    
    Args:
        work_dir: Directory containing epoch results
        expected_files: List of expected files per epoch
        
    Returns:
        ProgressInfo dict with completed_epochs, progress_percentage, early_stopped, early_stopped_at_epoch
    """
    # Try fast path: read progress.json
    progress_file = os.path.join(work_dir, "progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
                return ProgressInfo(**data)  # Return ProgressInfo dataclass instance
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise RuntimeError(f"Failed to load progress file {progress_file}: {e}") from e
    
    # Slow path: re-compute and create progress.json
    return _compute_and_cache_progress(work_dir, expected_files)


# ============================================================================
# PROGRESS CALCULATION FUNCTIONS
# ============================================================================

def _compute_and_cache_progress(work_dir: str, expected_files: List[str]) -> ProgressInfo:
    """Compute progress and create/update progress.json file."""
    # Count completed epochs (original logic)
    completed_epochs = 0
    while True:
        if not check_epoch_finished(
            epoch_dir=os.path.join(work_dir, f"epoch_{completed_epochs}"),
            expected_files=expected_files,
            check_load=False,
        ):
            break
        completed_epochs += 1
    
    # Load config to detect early stopping
    config_path = get_config(work_dir)
    config = load_config(config_path)
    tot_epochs = config['epochs']
    early_stopping_config = config.get('early_stopping')
    
    early_stopped = False
    early_stopped_at_epoch = None
    
    if early_stopping_config and completed_epochs < tot_epochs:
        # Early stopping is configured and we have fewer epochs than expected
        # Detect if early stopping was triggered
        early_stopped, early_stopped_at_epoch = _detect_early_stopping(
            work_dir, expected_files, config, completed_epochs
        )
    
    # Create progress.json for future caching
    progress_data = ProgressInfo(
        completed_epochs=completed_epochs,
        progress_percentage=100.0 if early_stopped else (completed_epochs / tot_epochs * 100.0),
        early_stopped=early_stopped,
        early_stopped_at_epoch=early_stopped_at_epoch,
        runner_type='trainer',  # session_progress is trainer-specific
        total_epochs=tot_epochs
    )
    
    progress_file = os.path.join(work_dir, "progress.json")
    save_json(progress_data, progress_file)
    
    # Return the ProgressInfo dataclass instance
    return progress_data


def _detect_early_stopping(work_dir: str, expected_files: List[str], config: dict, completed_epochs: int) -> Tuple[bool, Optional[int]]:
    """Detect if early stopping was triggered."""
    # Build metric and early stopping objects to detect early stopping
    metric = build_from_config(config['metric']) if config.get('metric') else None
    
    if metric is None:
        return False, None
        
    early_stopping_config = config['early_stopping']
    
    early_stopping = build_from_config(
        config=early_stopping_config,
        work_dir=work_dir,
        tot_epochs=config['epochs'],
        metric=metric,
        expected_files=expected_files,
        logger=None
    )
    
    # Check if early stopping would be triggered at the last completed epoch
    if completed_epochs > early_stopping.patience:
        last_epoch = completed_epochs - 1
        if early_stopping.was_triggered_at_epoch(last_epoch):
            return True, last_epoch + 1
    
    return False, None


# ============================================================================
# FILE CHECKING UTILITIES
# ============================================================================

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
