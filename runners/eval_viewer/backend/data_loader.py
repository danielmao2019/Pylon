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
    max_epoch: int
    metrics: Set[str]
    dataset_class: str
    dataset_type: DatasetType
    score_map: np.ndarray  # Shape: (N, C, H, W) where N=epochs, C=metrics, H=W=sqrt(n_datapoints)


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


def get_max_epoch(epoch_dirs: List[str]) -> int:
    """Get the maximum epoch number from a list of epoch directories.
    
    Args:
        epoch_dirs: List of epoch directory paths
        
    Returns:
        Maximum epoch number
    """
    return max(int(os.path.basename(d).split("_")[1]) for d in epoch_dirs)


def extract_metrics_from_scores(scores_file: str) -> Set[str]:
    """Extract metric names including sub-metrics from a validation scores file.
    
    Args:
        scores_file: Path to validation scores file
        
    Returns:
        Set of metric names including sub-metrics (e.g., class_tp[0])
        
    Raises:
        ValueError: If scores file format is invalid
    """
    with open(scores_file, "r") as f:
        scores = json.load(f)
        assert isinstance(scores, dict), f"Invalid scores format in {scores_file}"
        assert scores.keys() == {'aggregated', 'per_datapoint'}, f"Invalid keys in {scores_file}"
        assert isinstance(scores['per_datapoint'], dict), f"Invalid per_datapoint format in {scores_file}"
        
        # Extract metrics from per_datapoint scores
        metrics = set()
        for key in scores['per_datapoint'].keys():
            sample = scores['per_datapoint'][key][0]
            if isinstance(sample, list):
                metrics.update(f"{key}[{i}]" for i in range(len(sample)))
            else:
                assert isinstance(sample, (float, int)), f"Invalid sample type in {scores_file}"
                metrics.add(key)
                
        return metrics


def get_metrics(epoch_dirs: List[str]) -> Set[str]:
    """Get set of metrics from validation scores files.
    
    Args:
        epoch_dirs: List of epoch directory paths
        
    Returns:
        Set of metric names including sub-metrics (e.g., class_tp[0])
        
    Raises:
        ValueError: If no validation scores found or metrics are inconsistent
    """
    metrics = None
    for epoch_dir in epoch_dirs:
        scores_file = os.path.join(epoch_dir, "validation_scores.json")
        assert os.path.exists(scores_file)
        
        current_metrics = extract_metrics_from_scores(scores_file)
        if metrics is None:
            metrics = current_metrics
        elif current_metrics != metrics:
            raise ValueError(f"Inconsistent metrics found in {scores_file}")
                
    return metrics


def get_score_map(epoch_dirs: List[str], metric_names: Set[str], force_reload: bool = False) -> np.ndarray:
    """Get score map array from validation scores files or cache.
    
    Args:
        epoch_dirs: List of epoch directory paths
        metric_names: Set of metric names including sub-metrics
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
    
    # Get number of datapoints from first epoch
    with open(os.path.join(epoch_dirs[0], "validation_scores.json"), "r") as f:
        scores = json.load(f)
        n_datapoints = len(scores['per_datapoint'][next(iter(scores['per_datapoint'].keys()))])
    
    # Calculate dimensions for square matrix
    side_length = int(np.ceil(np.sqrt(n_datapoints)))
    
    # Initialize score map with NaN's
    n_epochs = len(epoch_dirs)
    n_metrics = len(metric_names)
    score_map = np.full((n_epochs, n_metrics, side_length, side_length), np.nan)
    
    # Sort metric names for consistent ordering
    sorted_metrics = sorted(metric_names)
    
    # Load all scores at once
    all_scores = []
    for epoch_dir in epoch_dirs:
        with open(os.path.join(epoch_dir, "validation_scores.json"), "r") as f:
            scores = json.load(f)
            epoch_scores = []
            for metric in sorted_metrics:
                if '[' in metric:
                    base_metric, idx_str = metric.split('[')
                    idx = int(idx_str.rstrip(']'))
                    metric_scores = np.array([float(score[idx]) for score in scores['per_datapoint'][base_metric]])
                else:
                    metric_scores = np.array([float(score) for score in scores['per_datapoint'][metric]])
                epoch_scores.append(metric_scores)
            all_scores.append(epoch_scores)
    
    # Convert to numpy array and reshape
    all_scores = np.array(all_scores)  # Shape: (n_epochs, n_metrics, n_datapoints)
    score_map.reshape(n_epochs, n_metrics, -1)[:, :, :n_datapoints] = all_scores
    
    # Save to cache
    np.save(cache_path, score_map)
    
    return score_map


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
    max_epoch = get_max_epoch(epoch_dirs)
    metrics = get_metrics(epoch_dirs)
    dataset_class, dataset_type = get_dataset_info(log_dir)
    score_map = get_score_map(epoch_dirs, metrics, force_reload)
    
    return LogDirInfo(
        max_epoch=max_epoch,
        metrics=metrics,
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
