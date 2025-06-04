from typing import List, Dict, Set, Tuple, NamedTuple
import os
import json
import numpy as np
from pathlib import Path
from data.viewer.managers.registry import get_dataset_type, DatasetType


class LogDirInfo(NamedTuple):
    """Information extracted from a log directory."""
    max_epoch: int
    metrics: Set[str]
    dataset_class: str
    dataset_type: DatasetType
    scores: Dict[int, Dict]  # epoch -> scores
    score_maps: np.ndarray  # (n_epochs, n_metrics, side_length, side_length)


def get_cache_path(log_dir: str) -> str:
    """Get the cache file path for a log directory.

    Args:
        log_dir: Path to log directory

    Returns:
        cache_path: Path to cache file
    """
    cache_dir = Path(__file__).parent.parent / "score_maps_cache"
    cache_dir.mkdir(exist_ok=True)

    # Use the last component of log_dir as the cache filename
    run_name = os.path.basename(os.path.normpath(log_dir))
    return str(cache_dir / f"{run_name}.npy")


def create_score_maps(scores: Dict[int, Dict], metrics: List[str]) -> np.ndarray:
    """Create score maps array from scores dictionary.

    Args:
        scores: Dictionary mapping epochs to scores
        metrics: List of metrics to include

    Returns:
        score_maps: numpy array of shape (N, C, H, W) where:
            N = number of epochs
            C = number of metrics
            H, W = dimensions of the square matrix
    """
    n_epochs = len(scores)
    n_datapoints = len(next(iter(scores[0]['per_datapoint'].values())))
    side_length = int(np.ceil(np.sqrt(n_datapoints)))

    # Initialize cache with NaN's
    score_maps = np.full((n_epochs, len(metrics), side_length, side_length), np.nan)

    # Fill cache for each epoch and metric
    for e in range(n_epochs):
        for c, metric in enumerate(metrics):
            if '[' in metric:
                base_metric, idx_str = metric.split('[')
                idx = int(idx_str.rstrip(']'))
                metric_scores = [float(score[idx]) for score in scores[e]['per_datapoint'][base_metric]]
            else:
                metric_scores = [float(score) for score in scores[e]['per_datapoint'][metric]]
            score_maps[e, c].flat[:n_datapoints] = metric_scores

    return score_maps


def extract_log_dir_info(log_dir: str, force_reload: bool = False) -> LogDirInfo:
    """Extract all necessary information from a log directory.
    
    Args:
        log_dir: Path to log directory
        force_reload: Whether to force reload of cached data
        
    Returns:
        LogDirInfo containing:
            - max_epoch: Number of completed epochs
            - metrics: Set of metric names
            - dataset_class: Name of dataset class
            - dataset_type: Type of dataset
            - scores: Dictionary mapping epochs to scores
            - score_maps: Cached score maps array
            
    Raises:
        AssertionError: If any validation fails
    """
    # Check directory exists
    assert os.path.isdir(log_dir), f"Directory does not exist: {log_dir}"
    
    # Get dataset info from config
    config_path = os.path.join(log_dir, "config.json")
    assert os.path.isfile(config_path), f"config.json not found in {log_dir}"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    assert 'val_dataset' in config, f"val_dataset not found in config.json in {log_dir}"
    assert 'class' in config['val_dataset'], f"dataset class not found in config.json in {log_dir}"
    
    dataset_class = config['val_dataset']['class']
    dataset_name = dataset_class.lower().replace('dataset', '')
    dataset_type = get_dataset_type(dataset_name)
    
    # Check if we can load from cache
    cache_path = get_cache_path(log_dir)
    if not force_reload and os.path.exists(cache_path):
        try:
            # Load cache
            cache_data = np.load(cache_path, allow_pickle=True)
            if isinstance(cache_data, np.ndarray) and cache_data.dtype == np.dtype('O'):
                # Cache contains both scores and score_maps
                scores = cache_data[0].item()
                score_maps = cache_data[1]
                metrics = set(cache_data[2])
                
                # Validate cache
                assert isinstance(scores, dict), "Invalid cache format: scores not a dict"
                assert isinstance(score_maps, np.ndarray), "Invalid cache format: score_maps not an array"
                assert isinstance(metrics, set), "Invalid cache format: metrics not a set"
                
                # Get max epoch from scores
                max_epoch = max(scores.keys())
                
                return LogDirInfo(max_epoch, metrics, dataset_class, dataset_type, scores, score_maps)
        except Exception as e:
            logger.warning(f"Failed to load cache for {log_dir}: {str(e)}")
    
    # Get epoch info and scores
    max_epoch = -1
    scores = {}
    metrics = None
    
    while True:
        epoch = max_epoch + 1
        epoch_dir = os.path.join(log_dir, f"epoch_{epoch}")
        scores_path = os.path.join(epoch_dir, "validation_scores.json")
        
        if not os.path.isdir(epoch_dir) or not os.path.isfile(scores_path):
            break
            
        # Load scores
        with open(scores_path, 'r') as f:
            epoch_scores = json.load(f)
            
        # Validate scores format
        assert isinstance(epoch_scores, dict), f"Invalid scores format in {scores_path}"
        assert epoch_scores.keys() == {'aggregated', 'per_datapoint'}, f"Invalid keys in {scores_path}"
        assert isinstance(epoch_scores['aggregated'], dict), f"Invalid aggregated format in {scores_path}"
        assert isinstance(epoch_scores['per_datapoint'], dict), f"Invalid per_datapoint format in {scores_path}"
        assert epoch_scores['aggregated'].keys() == epoch_scores['per_datapoint'].keys(), \
            f"Invalid keys in {scores_path}"
            
        # Extract metrics from first epoch
        if metrics is None:
            metrics = set()
            for key in epoch_scores['per_datapoint'].keys():
                sample = epoch_scores['per_datapoint'][key][0]
                if isinstance(sample, list):
                    metrics.update(f"{key}[{i}]" for i in range(len(sample)))
                else:
                    metrics.add(key)
                    
        # Validate metrics are consistent
        current_metrics = set()
        for key in epoch_scores['per_datapoint'].keys():
            sample = epoch_scores['per_datapoint'][key][0]
            if isinstance(sample, list):
                current_metrics.update(f"{key}[{i}]" for i in range(len(sample)))
            else:
                current_metrics.add(key)
        assert current_metrics == metrics, f"Inconsistent metrics in {scores_path}"
        
        scores[epoch] = epoch_scores
        max_epoch = epoch
        
    assert max_epoch >= 0, f"No completed epochs in {log_dir}"
    assert metrics is not None and len(metrics) > 0, f"No metrics found in {log_dir}"
    
    # Create score maps
    score_maps = create_score_maps(scores, sorted(list(metrics)))
    
    # Save to cache
    try:
        np.save(cache_path, np.array([scores, score_maps, metrics], dtype=object))
    except Exception as e:
        logger.warning(f"Failed to save cache for {log_dir}: {str(e)}")
    
    return LogDirInfo(max_epoch, metrics, dataset_class, dataset_type, scores, score_maps)


def initialize_log_dirs(log_dirs: List[str], force_reload: bool = False) -> Tuple[int, Set[str], DatasetType, Dict[str, LogDirInfo]]:
    """Initialize and validate all log directories.
    
    This function:
    1. Extracts all necessary information from each log directory
    2. Validates that all directories use the same dataset type
    3. Validates that all directories have consistent metrics
    4. Returns the common max epoch, metrics, dataset type, and all log directory info
    
    Args:
        log_dirs: List of paths to log directories
        force_reload: Whether to force reload of cached data
        
    Returns:
        Tuple containing:
            - max_epoch: Maximum epoch index where all runs have completed training
            - metrics: Set of common metric names
            - dataset_type: Type of dataset being used
            - log_dir_infos: Dictionary mapping log directories to their info
            
    Raises:
        AssertionError: If any validation fails
    """
    assert len(log_dirs) > 0, "No log directories provided"
    
    # Extract info from all log directories
    log_dir_infos = {}
    dataset_types = set()
    all_metrics = None
    
    for log_dir in log_dirs:
        info = extract_log_dir_info(log_dir, force_reload)
        log_dir_infos[log_dir] = info
        dataset_types.add(info.dataset_type)
        
        if all_metrics is None:
            all_metrics = info.metrics
        else:
            assert info.metrics == all_metrics, f"Inconsistent metrics in {log_dir}"
            
    # Validate all directories use the same dataset type
    assert len(dataset_types) == 1, f"Multiple dataset types found: {dataset_types}"
    dataset_type = dataset_types.pop()
    
    # Get common max epoch
    max_epoch = min(info.max_epoch for info in log_dir_infos.values())
    
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
