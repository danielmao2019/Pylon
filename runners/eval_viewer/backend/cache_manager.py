from typing import List, Dict
import os
import numpy as np
from pathlib import Path

from .data_loader import validate_log_directories, get_common_metrics, load_validation_scores, extract_metric_scores


def get_cache_path(log_dir: str) -> str:
    """
    Gets the cache file path for a log directory.

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


def create_score_maps_cache(log_dir: str, metrics: List[str]) -> np.ndarray:
    """
    Creates and returns the score maps cache for a log directory.

    Args:
        log_dir: Path to log directory
        metrics: List of metrics to cache

    Returns:
        score_maps: numpy array of shape (N, C, H, W) where:
            N = number of epochs
            C = number of metrics
            H, W = dimensions of the square matrix
    """
    # Count number of epochs
    epoch = 0
    while os.path.exists(os.path.join(log_dir, f"epoch_{epoch}", "validation_scores.json")):
        epoch += 1
    n_epochs = epoch

    # Get dimensions from first epoch
    scores = load_validation_scores(log_dir, 0)
    n_datapoints = len(next(iter(scores['per_datapoint'].values())))
    side_length = int(np.ceil(np.sqrt(n_datapoints)))

    # Initialize cache with NaN's
    score_maps = np.full((n_epochs, len(metrics), side_length, side_length), np.nan)

    # Fill cache for each epoch and metric
    for e in range(n_epochs):
        scores = load_validation_scores(log_dir, e)
        for c, metric in enumerate(metrics):
            metric_scores = extract_metric_scores(scores, metric)
            score_maps[e, c].flat[:n_datapoints] = metric_scores

    return score_maps


def load_or_create_cache(log_dirs: List[str], force_reload: bool = False) -> Dict[str, np.ndarray]:
    """
    Loads or creates the score maps cache for all log directories.

    Args:
        log_dirs: List of paths to log directories
        force_reload: Whether to force recreation of cache

    Returns:
        caches: Dictionary mapping log directory to score maps array
    """
    # Validate directories and get metrics
    validate_log_directories(log_dirs)
    metrics = sorted(list(get_common_metrics(log_dirs)))

    caches = {}
    for log_dir in log_dirs:
        cache_path = get_cache_path(log_dir)

        if force_reload or not os.path.exists(cache_path):
            # Create and save cache
            score_maps = create_score_maps_cache(log_dir, metrics)
            np.save(cache_path, score_maps)
        else:
            # Load existing cache
            score_maps = np.load(cache_path)

        caches[log_dir] = score_maps

    return caches
