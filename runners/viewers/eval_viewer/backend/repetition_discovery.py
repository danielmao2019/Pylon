"""
Helper functions for discovering and aggregating experiment repetitions.
"""
from typing import List, Dict, Set, Tuple, Any, Optional
import os
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from runners.viewers.eval_viewer.backend.initialization import LogDirInfo, extract_log_dir_info
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentGroup:
    """Information about a group of experiment repetitions."""
    base_path: str  # Base experiment path without _run_x suffix
    experiment_name: str  # Name of the experiment (e.g., "ICP", "RANSAC_FPFH")
    repetition_paths: List[str]  # List of valid repetition paths
    num_repetitions: int  # Number of valid repetitions found
    
    def get_log_dir_infos(self, force_reload: bool = False) -> List[LogDirInfo]:
        """Load LogDirInfo for all repetitions."""
        log_dir_infos = []
        for rep_path in self.repetition_paths:
            try:
                info = extract_log_dir_info(rep_path, force_reload=force_reload)
                log_dir_infos.append(info)
            except Exception as e:
                logger.warning(f"Failed to load repetition {rep_path}: {e}")
        return log_dir_infos


def _extract_base_path(log_dir: str) -> Tuple[str, Optional[int]]:
    """Extract base path and run number from log directory.
    
    Args:
        log_dir: Path to log directory (may or may not end with _run_x)
        
    Returns:
        Tuple of (base_path, run_number) where run_number is None if no _run_x suffix
        
    Examples:
        "/logs/kitti/ICP_run_0" -> ("/logs/kitti/ICP", 0)
        "/logs/kitti/ICP" -> ("/logs/kitti/ICP", None)
    """
    # Normalize path to remove trailing slashes
    log_dir = os.path.normpath(log_dir)
    
    # Check if path ends with _run_x pattern
    match = re.search(r'_run_(\d+)$', log_dir)
    if match:
        run_number = int(match.group(1))
        base_path = log_dir[:match.start()]
        return base_path, run_number
    else:
        return log_dir, None


def _discover_repetitions(base_path: str, existing_runs: Set[int], max_runs: int = 10) -> List[str]:
    """Discover all valid repetition directories for a given base path.
    
    Args:
        base_path: Base experiment path without _run_x suffix
        existing_runs: Set of run numbers that were explicitly provided
        max_runs: Maximum number of runs to search for
        
    Returns:
        List of valid repetition directory paths, sorted by run number
    """
    valid_repetitions = []
    
    # Get parent directory to search in
    parent_dir = os.path.dirname(base_path)
    if not os.path.exists(parent_dir):
        return valid_repetitions
    
    # Search for _run_x directories
    for run_idx in range(max_runs):
        candidate_path = f"{base_path}_run_{run_idx}"
        
        # Check if directory exists 
        if os.path.exists(candidate_path):
            # During discovery, we're less strict - just check if it looks like a log directory
            # Actual validation will happen later when LogDirInfo is extracted
            valid_repetitions.append(candidate_path)
            logger.info(f"Found potential repetition: {candidate_path}")
    
    return sorted(valid_repetitions)


def discover_experiment_groups(log_dirs: List[str]) -> List[ExperimentGroup]:
    """Discover experiment groups with repetitions from provided log directories.
    
    Args:
        log_dirs: List of log directory paths provided by user
        
    Returns:
        List of ExperimentGroup objects, each containing discovered repetitions
        
    Examples:
        Input: ["/logs/kitti/ICP_run_0", "/logs/kitti/RANSAC_FPFH_run_1"]
        Output: [
            ExperimentGroup(base_path="/logs/kitti/ICP", repetition_paths=["/logs/kitti/ICP_run_0", "/logs/kitti/ICP_run_1", "/logs/kitti/ICP_run_2"]),
            ExperimentGroup(base_path="/logs/kitti/RANSAC_FPFH", repetition_paths=["/logs/kitti/RANSAC_FPFH_run_0", "/logs/kitti/RANSAC_FPFH_run_1", "/logs/kitti/RANSAC_FPFH_run_2"])
        ]
    """
    # Input validation following CLAUDE.md fail-fast patterns
    assert log_dirs is not None, "log_dirs must not be None"
    assert isinstance(log_dirs, list), f"log_dirs must be list, got {type(log_dirs)}"
    assert len(log_dirs) > 0, f"log_dirs must not be empty"
    assert all(isinstance(log_dir, str) for log_dir in log_dirs), f"All log_dirs must be strings, got {log_dirs}"
    
    # Group by base path
    base_path_to_runs = {}
    for log_dir in log_dirs:
        base_path, run_number = _extract_base_path(log_dir)
        
        # Handle case where user provides path without _run_x suffix
        if run_number is None:
            # Treat as a base path and search for repetition directories (_run_x pattern)
            discovered_repetitions = _discover_repetitions(base_path, set(), max_runs=10)
            if discovered_repetitions:
                # Found repetitions - add them to the repetition group
                for rep_path in discovered_repetitions:
                    rep_base_path, rep_run_number = _extract_base_path(rep_path)
                    if rep_base_path not in base_path_to_runs:
                        base_path_to_runs[rep_base_path] = set()
                    base_path_to_runs[rep_base_path].add(rep_run_number)
                logger.info(f"Discovered {len(discovered_repetitions)} repetitions for base path: {base_path}")
            else:
                logger.warning(f"No repetitions found for base path: {base_path}")
        else:
            # Add to base path groups
            if base_path not in base_path_to_runs:
                base_path_to_runs[base_path] = set()
            base_path_to_runs[base_path].add(run_number)
    
    # Create repetition groups for each base path
    experiment_groups = []
    for base_path, existing_runs in base_path_to_runs.items():
        # Discover all repetitions for this base path
        discovered_repetitions = _discover_repetitions(base_path, existing_runs, max_runs=10)
        
        if discovered_repetitions:
            experiment_name = os.path.basename(base_path)
            group = ExperimentGroup(
                base_path=base_path,
                experiment_name=experiment_name,
                repetition_paths=discovered_repetitions,
                num_repetitions=len(discovered_repetitions)
            )
            experiment_groups.append(group)
            logger.info(f"Created experiment group '{experiment_name}' with {len(discovered_repetitions)} repetitions")
    
    assert len(experiment_groups) > 0, f"No valid experiment groups discovered from log_dirs: {log_dirs}"
    return experiment_groups


def aggregate_log_dir_infos(log_dir_infos: List[LogDirInfo]) -> LogDirInfo:
    """Aggregate multiple LogDirInfo objects across repetitions.
    
    Args:
        log_dir_infos: List of LogDirInfo objects from different repetitions
        
    Returns:
        Aggregated LogDirInfo with mean and std across repetitions
        
    Raises:
        AssertionError: If log_dir_infos have inconsistent structure
    """
    # Input validation following CLAUDE.md fail-fast patterns
    assert log_dir_infos is not None, "log_dir_infos must not be None"
    assert isinstance(log_dir_infos, list), f"log_dir_infos must be list, got {type(log_dir_infos)}"
    assert len(log_dir_infos) > 0, f"log_dir_infos must not be empty"
    assert all(isinstance(info, LogDirInfo) for info in log_dir_infos), f"All items must be LogDirInfo, got {[type(info) for info in log_dir_infos]}"
    
    if len(log_dir_infos) == 1:
        # No aggregation needed for single repetition
        return log_dir_infos[0]
    
    # Validate consistency across repetitions
    first_info = log_dir_infos[0]
    
    # Check consistent structure
    assert all(info.num_epochs == first_info.num_epochs for info in log_dir_infos), \
        f"Inconsistent num_epochs: {[info.num_epochs for info in log_dir_infos]}"
    assert all(info.metric_names == first_info.metric_names for info in log_dir_infos), \
        f"Inconsistent metric_names: {[info.metric_names for info in log_dir_infos]}"
    assert all(info.num_datapoints == first_info.num_datapoints for info in log_dir_infos), \
        f"Inconsistent num_datapoints: {[info.num_datapoints for info in log_dir_infos]}"
    assert all(info.dataset_class == first_info.dataset_class for info in log_dir_infos), \
        f"Inconsistent dataset_class: {[info.dataset_class for info in log_dir_infos]}"
    assert all(info.dataset_type == first_info.dataset_type for info in log_dir_infos), \
        f"Inconsistent dataset_type: {[info.dataset_type for info in log_dir_infos]}"
    assert all(info.runner_type == first_info.runner_type for info in log_dir_infos), \
        f"Inconsistent runner_type: {[info.runner_type for info in log_dir_infos]}"
    
    # Stack score maps and aggregated scores across repetitions
    score_maps = np.stack([info.score_map for info in log_dir_infos], axis=0)  # Shape: (R, N, C, H, W) or (R, C, H, W)
    aggregated_scores_array = np.stack([info.aggregated_scores for info in log_dir_infos], axis=0)  # Shape: (R, N, C) or (R, C)
    
    # Compute mean and std across repetitions (axis=0)
    score_map_mean = np.mean(score_maps, axis=0)
    score_map_std = np.std(score_maps, axis=0, ddof=1) if len(log_dir_infos) > 1 else np.zeros_like(score_map_mean)
    
    aggregated_scores_mean = np.mean(aggregated_scores_array, axis=0)
    aggregated_scores_std = np.std(aggregated_scores_array, axis=0, ddof=1) if len(log_dir_infos) > 1 else np.zeros_like(aggregated_scores_mean)
    
    # Create new LogDirInfo with aggregated data
    # Use the mean values for score_map and aggregated_scores
    # Store std values in additional attributes if needed in the future
    aggregated_info = LogDirInfo(
        num_epochs=first_info.num_epochs,
        metric_names=first_info.metric_names,
        num_datapoints=first_info.num_datapoints,
        score_map=score_map_mean,
        aggregated_scores=aggregated_scores_mean,
        dataset_class=first_info.dataset_class,
        dataset_type=first_info.dataset_type,
        dataset_cfg=first_info.dataset_cfg,
        dataloader_cfg=first_info.dataloader_cfg,
        runner_type=first_info.runner_type,
    )
    
    # Add std information as additional attributes (for potential future use)
    aggregated_info.score_map_std = score_map_std
    aggregated_info.aggregated_scores_std = aggregated_scores_std
    aggregated_info.num_repetitions = len(log_dir_infos)
    
    logger.info(f"Aggregated {len(log_dir_infos)} repetitions with mean and std")
    return aggregated_info