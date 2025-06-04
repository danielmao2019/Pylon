from typing import List, Dict, Set, Tuple, NamedTuple
import os
import json
import numpy as np
from pathlib import Path
from data.viewer.managers.registry import get_dataset_type, DatasetType
import logging

logger = logging.getLogger(__name__)


class LogDirInfo(NamedTuple):
    """Information extracted from a log directory."""
    num_epochs: int
    metric_names: Set[str]
    dataset_class: str
    dataset_type: DatasetType
    score_map: np.ndarray  # Shape: (N, C, H, W) where N=epochs, C=metrics, H=W=sqrt(n_datapoints)


def get_score_map_epoch_metric(scores_file: str, metric_name: str) -> Tuple[int, np.ndarray]:
    """Get score map for a single epoch and metric.
    
    Args:
        scores_file: Path to validation scores file
        metric_name: Name of metric (including sub-metrics)
        
    Returns:
        Tuple of (n_datapoints, score_map) where score_map has shape (H, W)
    """
    with open(scores_file, "r") as f:
        scores = json.load(f)
    
    if '[' in metric_name:
        base_metric, idx_str = metric_name.split('[')
        idx = int(idx_str.rstrip(']'))
        metric_scores = np.array([float(score[idx]) for score in scores['per_datapoint'][base_metric]])
    else:
        metric_scores = np.array([float(score) for score in scores['per_datapoint'][metric_name]])
    
    n_datapoints = len(metric_scores)
    side_length = int(np.ceil(np.sqrt(n_datapoints)))
    score_map = np.full((side_length, side_length), np.nan)
    score_map.flat[:n_datapoints] = metric_scores
    
    return n_datapoints, score_map


def get_metric_names(scores_dict: dict) -> Set[str]:
    """Extract metric names including sub-metrics from scores dictionary.
    
    Args:
        scores_dict: Dictionary containing per_datapoint scores
        
    Returns:
        Set of metric names including sub-metrics (e.g., class_tp[0])
    """
    metrics = set()
    for key in scores_dict.keys():
        sample = scores_dict[key][0]
        if isinstance(sample, list):
            metrics.update(f"{key}[{i}]" for i in range(len(sample)))
        else:
            assert isinstance(sample, (float, int)), f"Invalid sample type for metric {key}"
            metrics.add(key)
    return metrics


def get_score_map_epoch(scores_file: str) -> Tuple[Set[str], np.ndarray]:
    """Get score map for a single epoch and all metrics.
    
    Args:
        scores_file: Path to validation scores file
        
    Returns:
        Tuple of (metric_names, score_map) where score_map has shape (C, H, W)
    """
    with open(scores_file, "r") as f:
        scores = json.load(f)
    assert isinstance(scores, dict), f"Invalid scores format in {scores_file}"
    assert scores.keys() == {'aggregated', 'per_datapoint'}, f"Invalid keys in {scores_file}"
    
    metric_names = get_metric_names(scores['per_datapoint'])
    assert metric_names == get_metric_names(scores['aggregated'])
    
    all_score_maps_epoch = [
        get_score_map_epoch_metric(scores_file, metric_name)
        for metric_name in metric_names
    ]
    assert all(score_map_epoch_metric[0] == all_score_maps_epoch[0][0] 
              for score_map_epoch_metric in all_score_maps_epoch)
    
    score_map_epoch = np.stack([score_map_epoch_metric[1] 
                              for score_map_epoch_metric in all_score_maps_epoch], axis=0)
    return metric_names, score_map_epoch


def get_epoch_dirs(log_dir: str) -> List[str]:
    """Get list of consecutive epoch directories in a log directory that have validation scores.
    
    Args:
        log_dir: Path to log directory
        
    Returns:
        List of consecutive epoch directory paths that have validation scores
        
    Raises:
        ValueError: If no valid epoch directories found or epochs are not consecutive
    """
    epoch_dirs = []
    epoch = 0
    
    while True:
        epoch_dir = os.path.join(log_dir, f"epoch_{epoch}")
        scores_file = os.path.join(epoch_dir, "validation_scores.json")
        
        if not os.path.exists(epoch_dir) or not os.path.exists(scores_file):
            break
            
        epoch_dirs.append(epoch_dir)
        epoch += 1
    
    if not epoch_dirs:
        raise ValueError(f"No epoch directories with validation scores found in {log_dir}")
        
    return epoch_dirs


def get_dataset_info(log_dir: str) -> Tuple[str, DatasetType]:
    """Get dataset class and type from config file.
    
    Args:
        log_dir: Path to log directory
        
    Returns:
        Tuple of (dataset_class, dataset_type)
        
    Raises:
        ValueError: If config file not found or invalid
    """
    config_file = os.path.join(log_dir, "config.json")
    if not os.path.exists(config_file):
        raise ValueError(f"Config file not found: {config_file}")
        
    with open(config_file, "r") as f:
        config = json.load(f)
        
    dataset_class = config.get("dataset", {}).get("class")
    if not dataset_class:
        raise ValueError(f"Dataset class not found in {config_file}")
        
    dataset_type = get_dataset_type(dataset_class)
    return dataset_class, dataset_type


def get_score_map(epoch_dirs: List[str], force_reload: bool = False) -> np.ndarray:
    """Get score map array from validation scores files or cache.
    
    Args:
        epoch_dirs: List of epoch directory paths
        force_reload: Whether to force reload from source files
        
    Returns:
        Score map array of shape (N, C, H, W) where:
            N = number of epochs
            C = number of metrics
            H, W = dimensions of the square matrix (H*W = n_datapoints)
            
    Raises:
        ValueError: If scores format is invalid or cache is corrupted
    """
    # Get cache path
    cache_dir = Path(__file__).parent.parent / "score_maps_cache"
    cache_dir.mkdir(exist_ok=True)
    run_name = os.path.basename(os.path.normpath(os.path.dirname(epoch_dirs[0])))
    cache_path = str(cache_dir / f"{run_name}.npy")
    
    # Try to load from cache first
    if not force_reload and os.path.exists(cache_path):
        score_map = np.load(cache_path)
        if not isinstance(score_map, np.ndarray) or score_map.ndim != 4:
            raise ValueError(f"Invalid cache format in {cache_path}")
        return score_map
    
    # Get score maps for all epochs
    all_score_maps = [
        get_score_map_epoch(os.path.join(epoch_dir, "validation_scores.json"))
        for epoch_dir in epoch_dirs
    ]
    assert all(score_map_epoch[0] == all_score_maps[0][0] for score_map_epoch in all_score_maps)
    
    # Stack all epoch score maps
    score_map = np.stack([score_map_epoch[1] for score_map_epoch in all_score_maps], axis=0)
    
    # Save to cache
    np.save(cache_path, score_map)
    
    return score_map


def extract_log_dir_info(log_dir: str, force_reload: bool = False) -> LogDirInfo:
    """Extract all information from a log directory.
    
    Args:
        log_dir: Path to log directory
        force_reload: Whether to force reload from source files
        
    Returns:
        LogDirInfo object containing all extracted information
        
    Raises:
        ValueError: If log directory is invalid or required files are missing
    """
    # Check if log directory exists
    if not os.path.exists(log_dir):
        raise ValueError(f"Log directory not found: {log_dir}")
    
    # Extract information from source files
    epoch_dirs = get_epoch_dirs(log_dir)
    num_epochs = len(epoch_dirs)
    metric_names, score_map = get_score_map(epoch_dirs, force_reload)
    dataset_class, dataset_type = get_dataset_info(log_dir)

    return LogDirInfo(
        num_epochs=num_epochs,
        metric_names=metric_names,
        dataset_class=dataset_class,
        dataset_type=dataset_type,
        score_map=score_map
    )


def initialize_log_dirs(log_dirs: List[str], force_reload: bool = False) -> Tuple[int, Set[str], DatasetType, Dict[str, LogDirInfo]]:
    """Initialize log directories and validate consistency.
    
    Args:
        log_dirs: List of paths to log directories
        force_reload: Whether to force reload from source files
        
    Returns:
        Tuple of (max_epoch, metrics, dataset_type, log_dir_infos)
        
    Raises:
        ValueError: If log directories are invalid or inconsistent
    """
    # Extract information from each log directory
    log_dir_infos = {}
    for log_dir in log_dirs:
        try:
            info = extract_log_dir_info(log_dir, force_reload)
            log_dir_infos[log_dir] = info
        except Exception as e:
            raise ValueError(f"Failed to process log directory {log_dir}: {e}")
    
    # Validate consistency
    if not log_dir_infos:
        raise ValueError("No valid log directories found")
        
    # Get common information
    max_epoch = max(info.max_epoch for info in log_dir_infos.values())
    all_metrics = set.intersection(*(info.metrics for info in log_dir_infos.values()))
    dataset_types = {info.dataset_type for info in log_dir_infos.values()}
    
    # Validate dataset type consistency
    if len(dataset_types) != 1:
        raise ValueError(f"All log directories must use the same dataset type. Found: {dataset_types}")
    dataset_type = dataset_types.pop()
    
    # Validate metrics consistency
    if not all_metrics:
        raise ValueError("No common metrics found across log directories")
        
    return max_epoch, all_metrics, dataset_type, log_dir_infos


def extract_metric_scores(scores: Dict, metric: str) -> List[float]:
    """Extracts scores for a specific metric from validation scores.

    Args:
        scores: Dictionary containing validation scores
        metric: Name of the metric to extract

    Returns:
        metric_scores: List of float scores for the specified metric

    Raises:
        AssertionError: If metric doesn't exist in scores
    """
    # Handle sub-metrics (e.g., class_tp[0])
    if '[' in metric:
        base_metric, idx_str = metric.split('[')
        idx = int(idx_str.rstrip(']'))
        assert base_metric in scores['per_datapoint'], f"Metric {base_metric} not found in scores"
        assert isinstance(scores['per_datapoint'][base_metric], list), f"Metric {base_metric} is not a list"
        assert idx < len(scores['per_datapoint'][base_metric]), f"Index {idx} out of range for {base_metric}"
        return [float(score[idx]) for score in scores['per_datapoint'][base_metric]]
    else:
        assert metric in scores['per_datapoint'], f"Metric {metric} not found in scores"
        return [float(score) for score in scores['per_datapoint'][metric]]
