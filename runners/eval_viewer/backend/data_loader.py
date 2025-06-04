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
    scores: Dict[str, Dict[str, float]]
    score_maps: Dict[str, Dict[str, Dict[str, float]]]



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


def get_cache_path(log_dir: str) -> str:
    """Get the path to the cache file for a log directory.
    
    Args:
        log_dir: Path to log directory
        
    Returns:
        Path to cache file
    """
    # Create cache directory in eval viewer
    cache_dir = Path(__file__).parent.parent / "score_maps_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Use the last component of log_dir as the cache filename
    run_name = os.path.basename(os.path.normpath(log_dir))
    return str(cache_dir / f"{run_name}.npy")


def get_scores(epoch_dirs: List[str], metrics: Set[str]) -> Dict[str, Dict[str, float]]:
    """Get scores for each epoch and metric.
    
    Args:
        epoch_dirs: List of epoch directory paths
        metrics: Set of metric names
        
    Returns:
        Dictionary mapping epochs to metric scores
    """
    scores = {}
    for epoch_dir in epoch_dirs:
        epoch = int(os.path.basename(epoch_dir).split("_")[1])
        scores_file = os.path.join(epoch_dir, "val_scores.json")
        
        if not os.path.exists(scores_file):
            continue
            
        with open(scores_file, "r") as f:
            epoch_scores = json.load(f)
            scores[epoch] = {metric: epoch_scores[metric] for metric in metrics}
    return scores


def get_score_maps(epoch_dirs: List[str], metrics: Set[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Get score maps for each epoch, metric and datapoint.
    
    Args:
        epoch_dirs: List of epoch directory paths
        metrics: Set of metric names
        
    Returns:
        Dictionary mapping epochs to metric score maps
    """
    score_maps = {}
    for epoch_dir in epoch_dirs:
        epoch = int(os.path.basename(epoch_dir).split("_")[1])
        score_maps_file = os.path.join(epoch_dir, "val_score_maps.json")
        
        if not os.path.exists(score_maps_file):
            continue
            
        with open(score_maps_file, "r") as f:
            epoch_score_maps = json.load(f)
            score_maps[epoch] = {
                metric: epoch_score_maps[metric] 
                for metric in metrics 
                if metric in epoch_score_maps
            }
    return score_maps


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
        
    # Try to load from cache first
    cache_path = get_cache_path(log_dir)
    if not force_reload and os.path.exists(cache_path):
        try:
            cache_data = np.load(cache_path, allow_pickle=True)
            if isinstance(cache_data, np.ndarray) and cache_data.dtype == np.dtype('O'):
                # Cache contains all LogDirInfo fields
                return LogDirInfo(
                    max_epoch=cache_data[0],
                    metrics=set(cache_data[1]),
                    dataset_class=cache_data[2],
                    dataset_type=cache_data[3],
                    scores=cache_data[4].item(),
                    score_maps=cache_data[5].item()
                )
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
    
    # Extract information from source files
    epoch_dirs = get_epoch_dirs(log_dir)
    max_epoch = get_max_epoch(epoch_dirs)
    metrics = get_metrics(epoch_dirs)
    dataset_class, dataset_type = get_dataset_info(log_dir)
    scores = get_scores(epoch_dirs, metrics)
    score_maps = get_score_maps(epoch_dirs, metrics)
    
    # Create LogDirInfo object
    info = LogDirInfo(
        max_epoch=max_epoch,
        metrics=metrics,
        dataset_class=dataset_class,
        dataset_type=dataset_type,
        scores=scores,
        score_maps=score_maps
    )
    
    # Cache the results
    try:
        np.save(cache_path, np.array([
            info.max_epoch,
            list(info.metrics),
            info.dataset_class,
            info.dataset_type,
            info.scores,
            info.score_maps
        ], dtype=object))
    except Exception as e:
        logger.warning(f"Failed to save cache to {cache_path}: {e}")
    
    return info


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
