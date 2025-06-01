from typing import List, Dict, Set, Tuple
import os
import json
import numpy as np
import glob


def validate_log_directories(log_dirs: List[str]) -> int:
    """
    Validates log directories and returns max epoch index.
    
    This function checks that:
    1. All provided log directories exist
    2. Each directory has epoch_0, epoch_1, etc.
    3. Each epoch directory has validation_scores.json
    4. All validation_scores.json have the same set of metrics
    5. Returns the maximum epoch index where all runs have completed training
    
    Args:
        log_dirs: List of paths to log directories
        
    Returns:
        max_epoch: Maximum epoch index where all runs have completed training
        
    Raises:
        AssertionError: If any validation fails
    """
    # Check all directories exist
    for log_dir in log_dirs:
        assert os.path.isdir(log_dir), f"Directory does not exist: {log_dir}"
    
    # Find max epoch for each run
    max_epochs = []
    for log_dir in log_dirs:
        idx = 0
        while True:
            epoch_dir = os.path.join(log_dir, f"epoch_{idx}")
            scores_path = os.path.join(epoch_dir, "validation_scores.json")
            
            if not os.path.isdir(epoch_dir) or not os.path.isfile(scores_path):
                break
                
            idx += 1
        
        assert idx > 0, f"No completed epochs in {log_dir}"
        max_epochs.append(idx - 1)
    
    # Return minimum max epoch (where all runs have completed)
    return min(max_epochs)


def get_common_metrics(log_dirs: List[str]) -> Set[str]:
    """
    Gets the set of common metrics across all validation scores files.
    
    Args:
        log_dirs: List of paths to log directories
        
    Returns:
        metrics: Set of metric names that are present in all validation scores files
        
    Raises:
        AssertionError: If no common metrics are found
    """
    # Get metrics from first run's first epoch
    first_log_dir = log_dirs[0]
    first_epoch_dir = os.path.join(first_log_dir, "epoch_0")
    first_scores_path = os.path.join(first_epoch_dir, "validation_scores.json")
    
    with open(first_scores_path, 'r') as f:
        first_scores = json.load(f)
    
    # Get all metrics from first run
    metrics = set()
    for key in first_scores['aggregated'].keys():
        if isinstance(first_scores['aggregated'][key], list):
            # Handle sub-metrics (e.g., class_tp[0], class_tp[1], etc.)
            for i in range(len(first_scores['aggregated'][key])):
                metrics.add(f"{key}[{i}]")
        else:
            metrics.add(key)
    
    # Verify metrics are consistent across all runs and epochs
    for log_dir in log_dirs:
        epoch_dirs = [d for d in os.listdir(log_dir) if d.startswith("epoch_")]
        for epoch_dir in epoch_dirs:
            scores_path = os.path.join(log_dir, epoch_dir, "validation_scores.json")
            with open(scores_path, 'r') as f:
                scores = json.load(f)
            
            # Get metrics from current file
            current_metrics = set()
            for key in scores['aggregated'].keys():
                if isinstance(scores['aggregated'][key], list):
                    for i in range(len(scores['aggregated'][key])):
                        current_metrics.add(f"{key}[{i}]")
                else:
                    current_metrics.add(key)
            
            # Verify metrics match
            assert current_metrics == metrics, f"Inconsistent metrics in {scores_path}"
    
    assert len(metrics) > 0, "No metrics found"
    return metrics


def load_validation_scores(log_dir: str, epoch: int) -> Dict:
    """
    Loads validation scores from a specific epoch.
    
    Args:
        log_dir: Path to log directory
        epoch: Epoch index
        
    Returns:
        scores: Dictionary containing validation scores
        
    Raises:
        AssertionError: If file doesn't exist or has invalid format
    """
    scores_path = os.path.join(log_dir, f"epoch_{epoch}", "validation_scores.json")
    assert os.path.isfile(scores_path), f"validation_scores.json not found at {scores_path}"
    
    with open(scores_path, 'r') as f:
        scores = json.load(f)
    
    assert isinstance(scores, dict), f"Invalid scores format in {scores_path}"
    assert 'aggregated' in scores and 'per_datapoint' in scores, f"Missing required keys in {scores_path}"
    
    return scores


def extract_metric_scores(scores: Dict, metric: str) -> List[float]:
    """
    Extracts scores for a specific metric from validation scores.
    
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
