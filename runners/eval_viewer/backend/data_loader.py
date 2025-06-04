from typing import List, Dict, Set, Tuple, NamedTuple
import os
import json
from data.viewer.managers.registry import get_dataset_type, DatasetType


class LogDirInfo(NamedTuple):
    """Information extracted from a log directory."""
    max_epoch: int
    metrics: Set[str]
    dataset_class: str
    dataset_type: DatasetType
    scores: Dict[int, Dict]  # epoch -> scores


def extract_log_dir_info(log_dir: str) -> LogDirInfo:
    """Extract all necessary information from a log directory.
    
    Args:
        log_dir: Path to log directory
        
    Returns:
        LogDirInfo containing:
            - max_epoch: Number of completed epochs
            - metrics: Set of metric names
            - dataset_class: Name of dataset class
            - dataset_type: Type of dataset
            - scores: Dictionary mapping epochs to scores
            
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
    
    return LogDirInfo(max_epoch, metrics, dataset_class, dataset_type, scores)


def initialize_log_dirs(log_dirs: List[str]) -> Tuple[int, Set[str], DatasetType, Dict[str, LogDirInfo]]:
    """Initialize and validate all log directories.
    
    This function:
    1. Extracts all necessary information from each log directory
    2. Validates that all directories use the same dataset type
    3. Validates that all directories have consistent metrics
    4. Returns the common max epoch, metrics, dataset type, and all log directory info
    
    Args:
        log_dirs: List of paths to log directories
        
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
        info = extract_log_dir_info(log_dir)
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
