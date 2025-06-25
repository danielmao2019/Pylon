from typing import List, Dict, Set, Tuple, NamedTuple, Any, Literal
import importlib.util
import os
import json
import numpy as np
import pickle
from pathlib import Path
from data.viewer.backend.backend import DatasetType, DATASET_GROUPS
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


def get_dataset_type(dataset_name: str) -> DatasetType:
    """Determine the dataset type based on the dataset name."""
    for dataset_type, datasets in DATASET_GROUPS.items():
        if dataset_name in datasets:
            return dataset_type
    raise ValueError(f"Unknown dataset type for dataset: {dataset_name}")


def detect_runner_type(log_dir: str) -> Literal['trainer', 'evaluator']:
    """Detect whether log directory contains BaseTrainer or BaseEvaluator results.
    
    Args:
        log_dir: Path to log directory
        
    Returns:
        'trainer' if directory contains epoch folders with validation_scores.json
        'evaluator' if directory contains evaluation_scores.json directly
        
    Raises:
        ValueError: If neither pattern is detected
    """
    # Check for BaseEvaluator pattern: evaluation_scores.json directly in log_dir
    evaluation_scores_path = os.path.join(log_dir, "evaluation_scores.json")
    if os.path.exists(evaluation_scores_path):
        return 'evaluator'
    
    # Check for BaseTrainer pattern: epoch folders with validation_scores.json
    epoch_0_dir = os.path.join(log_dir, "epoch_0")
    validation_scores_path = os.path.join(epoch_0_dir, "validation_scores.json")
    if os.path.exists(epoch_0_dir) and os.path.exists(validation_scores_path):
        return 'trainer'
    
    raise ValueError(f"Unable to detect log directory runner type for {log_dir}. "
                   f"Expected either 'evaluation_scores.json' in root or 'epoch_*/validation_scores.json' structure.")


class LogDirInfo(NamedTuple):
    """Information extracted from a log directory."""
    num_epochs: int
    metric_names: Set[str]
    num_datapoints: int  # Number of datapoints in the dataset
    score_map: np.ndarray  # Shape: (N, C, H, W) for trainer or (C, H, W) for evaluator
    aggregated_scores: np.ndarray  # Shape: (N, C) for trainer or (C,) for evaluator
    dataset_class: str
    dataset_type: DatasetType  # 2d_change_detection, 3d_change_detection, point_cloud_registration, etc.
    dataset_cfg: Dict[str, Any]
    dataloader_cfg: Dict[str, Any]
    runner_type: Literal['trainer', 'evaluator']  # Whether results come from BaseTrainer or BaseEvaluator


def get_score_map_epoch_metric(scores_file: str, metric_name: str) -> Tuple[int, np.ndarray, float]:
    """Get score map for a single epoch and metric.

    Args:
        scores_file: Path to validation scores file
        metric_name: Name of metric (including sub-metrics)

    Returns:
        Tuple of (num_datapoints, score_map, aggregated_score) where score_map has shape (H, W)
    """
    with open(scores_file, "r") as f:
        scores = json.load(f)

    if '[' in metric_name:
        base_metric, idx_str = metric_name.split('[')
        idx = int(idx_str.rstrip(']'))
        assert base_metric in scores['per_datapoint'], f"Metric {base_metric} not found in scores"
        assert isinstance(scores['per_datapoint'][base_metric], list), f"Metric {base_metric} is not a list"
        assert idx < len(scores['per_datapoint'][base_metric]), f"Index {idx} out of range for {base_metric}"
        per_datapoint_scores = np.array([float(score[idx]) for score in scores['per_datapoint'][base_metric]])
        aggregated_score = float(scores['aggregated'][base_metric][idx])
    else:
        assert metric_name in scores['per_datapoint'], f"Metric {metric_name} not found in scores"
        per_datapoint_scores = np.array([float(score) for score in scores['per_datapoint'][metric_name]])
        aggregated_score = float(scores['aggregated'][metric_name])

    assert per_datapoint_scores.ndim == 1, f"Per-datapoint scores {per_datapoint_scores} is not 1D"
    assert isinstance(aggregated_score, float), f"Aggregated score {aggregated_score} is not a float"

    num_datapoints = len(per_datapoint_scores)
    side_length = int(np.ceil(np.sqrt(num_datapoints)))
    score_map = np.full((side_length, side_length), np.nan)
    score_map.flat[:num_datapoints] = per_datapoint_scores

    return num_datapoints, score_map, aggregated_score


def get_metric_names_aggregated(scores_dict: dict) -> List[str]:
    """Extract metric names from aggregated scores dictionary.

    Args:
        scores_dict: Dictionary containing aggregated scores

    Returns:
        List of metric names
    """
    metrics = set()
    for key in scores_dict.keys():
        if isinstance(scores_dict[key], list):
            metrics.update(f"{key}[{i}]" for i in range(len(scores_dict[key])))
        else:
            assert isinstance(scores_dict[key], (float, int)), f"Invalid sample type for metric {key}"
            metrics.add(key)
    metrics = list(sorted(metrics))
    return metrics


def get_metric_names_per_datapoint(scores_dict: dict) -> List[str]:
    """Extract metric names including sub-metrics from scores dictionary.

    Args:
        scores_dict: Dictionary containing per_datapoint scores

    Returns:
        List of metric names including sub-metrics (e.g., class_tp[0])
    """
    metrics = set()
    for key in scores_dict.keys():
        assert isinstance(scores_dict[key], list), f"Invalid scores format in {scores_dict}"
        sample = scores_dict[key][0]
        if isinstance(sample, list):
            metrics.update(f"{key}[{i}]" for i in range(len(sample)))
        else:
            assert isinstance(sample, (float, int)), f"Invalid sample type for metric {key}"
            metrics.add(key)
    metrics = list(sorted(metrics))
    return metrics


def get_score_map_epoch(scores_file: str) -> Tuple[List[str], int, np.ndarray, np.ndarray]:
    """Get score map for a single epoch and all metrics.

    Args:
        scores_file: Path to validation scores file

    Returns:
        Tuple of (metric_names, num_datapoints, score_map, aggregated_scores)
    """
    with open(scores_file, "r") as f:
        scores = json.load(f)
    assert isinstance(scores, dict), f"Invalid scores format in {scores_file}"
    assert scores.keys() == {'aggregated', 'per_datapoint'}, f"Invalid keys in {scores_file}"

    metric_names = get_metric_names_aggregated(scores['aggregated'])
    assert metric_names == get_metric_names_per_datapoint(scores['per_datapoint'])

    all_score_maps_epoch = [
        get_score_map_epoch_metric(scores_file, metric_name)
        for metric_name in metric_names
    ]

    # Validate that all metrics have the same number of datapoints
    num_datapoints = all_score_maps_epoch[0][0]
    assert all(
        score_map_epoch_metric[0] == num_datapoints
        for score_map_epoch_metric in all_score_maps_epoch
    ), f"""{list(zip(
        metric_names,
        [score_map_epoch_metric[0] for score_map_epoch_metric in all_score_maps_epoch],
    ))}"""

    score_map_epoch = np.stack([
        score_map_epoch_metric[1] for score_map_epoch_metric in all_score_maps_epoch
    ], axis=0)
    aggregated_scores_epoch = np.array([
        score_map_epoch_metric[2] for score_map_epoch_metric in all_score_maps_epoch
    ])
    return metric_names, num_datapoints, score_map_epoch, aggregated_scores_epoch


def get_evaluator_score_map(scores_file: str) -> Tuple[List[str], int, np.ndarray, np.ndarray]:
    """Get score map for BaseEvaluator results (single evaluation).
    
    Args:
        scores_file: Path to evaluation_scores.json file
        
    Returns:
        Tuple of (metric_names, num_datapoints, score_map, aggregated_scores)
        score_map has shape (C, H, W) where C=metrics, H=W=sqrt(num_datapoints)
        aggregated_scores has shape (C,) where C=metrics
    """
    with open(scores_file, "r") as f:
        scores = json.load(f)
    assert isinstance(scores, dict), f"Invalid scores format in {scores_file}"
    assert scores.keys() == {'aggregated', 'per_datapoint'}, f"Invalid keys in {scores_file}"

    metric_names = get_metric_names_aggregated(scores['aggregated'])
    assert metric_names == get_metric_names_per_datapoint(scores['per_datapoint'])

    all_score_maps_metric = [
        get_score_map_epoch_metric(scores_file, metric_name)
        for metric_name in metric_names
    ]

    # Validate that all metrics have the same number of datapoints
    num_datapoints = all_score_maps_metric[0][0]
    assert all(
        score_map_metric[0] == num_datapoints
        for score_map_metric in all_score_maps_metric
    ), f"""{list(zip(
        metric_names,
        [score_map_metric[0] for score_map_metric in all_score_maps_metric],
    ))}"""

    # Create score_map with shape (C, H, W) instead of (N, C, H, W)
    score_map = np.stack([
        score_map_metric[1] for score_map_metric in all_score_maps_metric
    ], axis=0)
    # Create aggregated_scores with shape (C,) instead of (N, C)
    aggregated_scores = np.array([
        score_map_metric[2] for score_map_metric in all_score_maps_metric
    ])
    
    return metric_names, num_datapoints, score_map, aggregated_scores


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


def get_score_map(epoch_dirs: List[str]) -> Tuple[List[str], int, np.ndarray, np.ndarray]:
    """Get score map array from validation scores files.

    Args:
        epoch_dirs: List of epoch directory paths

    Returns:
        Tuple of (metric_names, num_datapoints, score_map, aggregated_scores)

    Raises:
        ValueError: If scores format is invalid
    """
    # Get score maps for all epochs in parallel while preserving order
    results = {}
    with ThreadPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(get_score_map_epoch, os.path.join(epoch_dir, "validation_scores.json")): idx
            for idx, epoch_dir in enumerate(epoch_dirs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    # Sort results by index to maintain order
    all_score_maps = [results[idx] for idx in range(len(epoch_dirs))]

    # Validate that all epochs have the same metric names
    metric_names = all_score_maps[0][0]
    assert all(score_map_epoch[0] == metric_names for score_map_epoch in all_score_maps), \
        f"Different metrics across epochs: {[score_map_epoch[0] for score_map_epoch in all_score_maps]}"

    # Validate that all epochs have the same number of datapoints
    num_datapoints = all_score_maps[0][1]
    assert all(score_map_epoch[1] == num_datapoints for score_map_epoch in all_score_maps), \
        f"Different number of datapoints across epochs: {[score_map_epoch[1] for score_map_epoch in all_score_maps]}"

    # Stack all epoch score maps
    score_map = np.stack([score_map_epoch[2] for score_map_epoch in all_score_maps], axis=0)
    aggregated_scores = np.stack([score_map_epoch[3] for score_map_epoch in all_score_maps], axis=0)
    assert score_map.shape[:2] == aggregated_scores.shape, \
        f"Score map and aggregated scores have different shapes: {score_map.shape} != {aggregated_scores.shape}"

    return metric_names, num_datapoints, score_map, aggregated_scores


def get_data_info(log_dir: str) -> Tuple[str, DatasetType, Dict[str, Any], Dict[str, Any]]:
    """Get dataset class and type from config file.

    Args:
        log_dir: Path to log directory

    Returns:
        Tuple of (dataset_class, dataset_type, dataset_cfg, dataloader_cfg)

    Raises:
        ValueError: If config file not found or invalid
    """
    dataset_class = log_dir.split("/")[-2]
    dataset_type = get_dataset_type(dataset_class)

    # Get the config file path
    config_file = os.path.join("./configs", os.path.relpath(log_dir, "./logs")) + ".py"
    assert os.path.isfile(config_file), f"Config file not found: {config_file}"

    # Load the config module
    spec = importlib.util.spec_from_file_location("config_file", config_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config

    # Extract dataset and dataloader configs based on runner type
    # BaseTrainer uses 'val_dataset', BaseEvaluator uses 'eval_dataset'
    if 'val_dataset' in config:
        dataset_cfg = config['val_dataset']
        dataloader_cfg = config['val_dataloader']
    elif 'eval_dataset' in config:
        dataset_cfg = config['eval_dataset']
        dataloader_cfg = config['eval_dataloader']
    else:
        raise ValueError(f"Config must contain either 'val_dataset' or 'eval_dataset', found keys: {list(config.keys())}")
    
    return dataset_class, dataset_type, dataset_cfg, dataloader_cfg


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

    # Get cache path
    cache_dir = Path(__file__).parent.parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    run_name = os.path.basename(os.path.normpath(log_dir))
    cache_path = str(cache_dir / f"{run_name}.pkl")

    # Try to load from cache first
    if not force_reload and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Detect runner type and extract information accordingly
    runner_type = detect_runner_type(log_dir)
    dataset_class, dataset_type, dataset_cfg, dataloader_cfg = get_data_info(log_dir)
    
    if runner_type == 'trainer':
        # BaseTrainer results: load from epoch folders
        epoch_dirs = get_epoch_dirs(log_dir)
        metric_names, num_datapoints, score_map, aggregated_scores = get_score_map(epoch_dirs)
        num_epochs = len(epoch_dirs)
    elif runner_type == 'evaluator':
        # BaseEvaluator results: load from evaluation_scores.json
        scores_file = os.path.join(log_dir, "evaluation_scores.json")
        metric_names, num_datapoints, score_map, aggregated_scores = get_evaluator_score_map(scores_file)
        num_epochs = 1  # BaseEvaluator produces only one evaluation result
    else:
        raise ValueError(f"Unknown runner type: {runner_type}")

    # Create LogDirInfo object
    info = LogDirInfo(
        num_epochs=num_epochs,
        metric_names=metric_names,
        num_datapoints=num_datapoints,
        score_map=score_map,
        aggregated_scores=aggregated_scores,
        dataset_class=dataset_class,
        dataset_type=dataset_type,
        dataset_cfg=dataset_cfg,
        dataloader_cfg=dataloader_cfg,
        runner_type=runner_type,
    )

    # Save to cache
    with open(cache_path, 'wb') as f:
        pickle.dump(info, f)

    return info


def initialize_log_dirs(log_dirs: List[str], force_reload: bool = False) -> Tuple[
    int, Set[str], int, Dict[str, Any], DatasetType, Dict[str, LogDirInfo],
]:
    """Initialize log directories and validate consistency.

    Args:
        log_dirs: List of paths to log directories
        force_reload: Whether to force reload from source files

    Returns:
        Tuple of (max_epoch, metrics, num_datapoints, dataset_cfg, dataset_type, log_dir_infos)

    Raises:
        ValueError: If log directories are invalid or inconsistent
    """
    # Extract information from each log directory in parallel
    log_dir_infos = {}
    with ThreadPoolExecutor() as executor:
        future_to_log_dir = {
            executor.submit(extract_log_dir_info, log_dir, force_reload): log_dir
            for log_dir in log_dirs
        }
        for future in as_completed(future_to_log_dir):
            log_dir = future_to_log_dir[future]
            log_dir_infos[log_dir] = future.result()

    # Get common information
    max_epochs = max(info.num_epochs for info in log_dir_infos.values())
    assert all(
        info.metric_names == list(log_dir_infos.values())[0].metric_names
        for info in log_dir_infos.values()
    ), f"""{list({
        key: info.metric_names
        for key, info in log_dir_infos.items()
    }.items())}"""
    metric_names = list(log_dir_infos.values())[0].metric_names
    assert all(info.dataset_class == list(log_dir_infos.values())[0].dataset_class for info in log_dir_infos.values())
    num_datapoints = list(log_dir_infos.values())[0].num_datapoints
    assert all(info.num_datapoints == num_datapoints for info in log_dir_infos.values())
    dataset_class = list(log_dir_infos.values())[0].dataset_class
    assert all(info.dataset_type == list(log_dir_infos.values())[0].dataset_type for info in log_dir_infos.values())
    dataset_type = list(log_dir_infos.values())[0].dataset_type
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
    # Map dataset types to their config directory paths
    dataset_type_to_dir = {
        'semseg': 'semantic_segmentation',
        '2dcd': 'change_detection',
        '3dcd': 'change_detection',
        'pcr': 'point_cloud_registration',
    }
    config_dir = dataset_type_to_dir[dataset_type]
    config_file = os.path.join(repo_root, "configs", "common", "datasets", config_dir, "val", f"{dataset_class}_data_cfg.py")
    spec = importlib.util.spec_from_file_location("config_file", config_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data_cfg = module.data_cfg
    dataset_cfg = data_cfg['val_dataset']

    return max_epochs, metric_names, num_datapoints, dataset_cfg, dataset_type, log_dir_infos
