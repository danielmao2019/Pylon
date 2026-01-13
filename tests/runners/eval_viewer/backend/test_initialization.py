import pytest
import numpy as np
import json
import os
import tempfile
from runners.viewers.eval_viewer.backend.initialization import (
    get_score_map_epoch_metric,
    get_metric_names_aggregated,
    get_metric_names_per_datapoint,
    get_score_map_epoch,
    get_evaluator_score_map,
    get_epoch_dirs,
    compute_per_metric_color_scales,
    LogDirInfo
)


# ==========================================
# Tests for get_metric_names_aggregated
# ==========================================

def test_get_metric_names_aggregated_simple_metrics():
    """Test get_metric_names_aggregated with simple scalar metrics."""
    scores_dict = {
        'metric1': 0.75,
        'metric2': 0.85,
        'metric3': 0.92
    }

    result = get_metric_names_aggregated(scores_dict)

    # Should return sorted list of metric names
    assert result == ['metric1', 'metric2', 'metric3']


def test_get_metric_names_aggregated_list_metrics():
    """Test get_metric_names_aggregated with list-valued metrics."""
    scores_dict = {
        'scalar_metric': 0.8,
        'class_metric': [0.7, 0.8, 0.9],
        'another_scalar': 0.6
    }

    result = get_metric_names_aggregated(scores_dict)

    # Should expand list metrics with indices and sort
    expected = ['another_scalar', 'class_metric[0]', 'class_metric[1]', 'class_metric[2]', 'scalar_metric']
    assert result == expected


def test_get_metric_names_aggregated_mixed_types():
    """Test get_metric_names_aggregated with mixed metric types."""
    scores_dict = {
        'single': 0.5,
        'multi': [0.1, 0.2, 0.3, 0.4],
        'another_single': 0.9
    }

    result = get_metric_names_aggregated(scores_dict)

    expected = ['another_single', 'multi[0]', 'multi[1]', 'multi[2]', 'multi[3]', 'single']
    assert result == expected


def test_get_metric_names_aggregated_empty_dict():
    """Test get_metric_names_aggregated with empty dictionary."""
    result = get_metric_names_aggregated({})
    assert result == []


def test_get_metric_names_aggregated_invalid_type():
    """Test get_metric_names_aggregated fails with invalid metric type."""
    scores_dict = {
        'valid_metric': 0.8,
        'invalid_metric': 'string_value'  # Invalid type
    }

    with pytest.raises(AssertionError, match="Invalid sample type for metric"):
        get_metric_names_aggregated(scores_dict)


# ==========================================
# Tests for get_metric_names_per_datapoint
# ==========================================

def test_get_metric_names_per_datapoint_simple_metrics():
    """Test get_metric_names_per_datapoint with simple scalar metrics."""
    scores_dict = {
        'metric1': [0.7, 0.8, 0.75],
        'metric2': [0.9, 0.85, 0.87]
    }

    result = get_metric_names_per_datapoint(scores_dict)

    assert result == ['metric1', 'metric2']


def test_get_metric_names_per_datapoint_list_metrics():
    """Test get_metric_names_per_datapoint with list-valued metrics."""
    scores_dict = {
        'scalar_metric': [0.8, 0.75, 0.82],
        'class_metric': [
            [0.7, 0.8, 0.9],
            [0.75, 0.85, 0.95],
            [0.72, 0.82, 0.92]
        ]
    }

    result = get_metric_names_per_datapoint(scores_dict)

    expected = ['class_metric[0]', 'class_metric[1]', 'class_metric[2]', 'scalar_metric']
    assert result == expected


def test_get_metric_names_per_datapoint_invalid_format():
    """Test get_metric_names_per_datapoint fails with invalid format."""
    scores_dict = {
        'metric1': [0.8, 0.9],
        'invalid_metric': 0.8  # Should be list
    }

    with pytest.raises(AssertionError, match="Invalid scores format"):
        get_metric_names_per_datapoint(scores_dict)


def test_get_metric_names_per_datapoint_invalid_sample_type():
    """Test get_metric_names_per_datapoint fails with invalid sample type."""
    scores_dict = {
        'valid_metric': [0.8, 0.9],
        'invalid_metric': ['string', 'values']  # Invalid sample type
    }

    with pytest.raises(AssertionError, match="Invalid sample type for metric"):
        get_metric_names_per_datapoint(scores_dict)


# ==========================================
# Tests for get_score_map_epoch_metric
# ==========================================

def test_get_score_map_epoch_metric_simple_metric(validation_scores_file):
    """Test get_score_map_epoch_metric with simple scalar metric."""
    num_datapoints, score_map, aggregated_score = get_score_map_epoch_metric(
        validation_scores_file, 'metric1'
    )

    assert num_datapoints == 5
    assert isinstance(score_map, np.ndarray)
    assert score_map.shape == (3, 3)  # ceil(sqrt(5)) = 3
    assert isinstance(aggregated_score, float)
    assert aggregated_score == 0.75

    # Check that first 5 values match the input scores
    expected_scores = [0.7, 0.8, 0.75, 0.72, 0.78]
    for i, expected in enumerate(expected_scores):
        assert score_map.flat[i] == expected

    # Check that remaining positions are NaN
    assert np.all(np.isnan(score_map.flat[5:]))


def test_get_score_map_epoch_metric_list_metric(validation_scores_file):
    """Test get_score_map_epoch_metric with list-valued metric."""
    num_datapoints, score_map, aggregated_score = get_score_map_epoch_metric(
        validation_scores_file, 'multi_metric[1]'
    )

    assert num_datapoints == 5
    assert isinstance(score_map, np.ndarray)
    assert score_map.shape == (3, 3)
    assert isinstance(aggregated_score, float)
    assert aggregated_score == 0.7  # Second element of [0.8, 0.7, 0.9]

    # Check that first 5 values match the expected scores (index 1 from each datapoint)
    expected_scores = [0.7, 0.72, 0.68, 0.74, 0.71]  # Index 1 from multi_metric lists
    for i, expected in enumerate(expected_scores):
        assert score_map.flat[i] == expected


def test_get_score_map_epoch_metric_nonexistent_metric(validation_scores_file):
    """Test get_score_map_epoch_metric fails with nonexistent metric."""
    with pytest.raises(AssertionError, match="Metric nonexistent not found in scores"):
        get_score_map_epoch_metric(validation_scores_file, 'nonexistent')


def test_get_score_map_epoch_metric_invalid_index(validation_scores_file):
    """Test get_score_map_epoch_metric fails with invalid list index."""
    with pytest.raises(AssertionError, match="Index 5 out of range"):
        get_score_map_epoch_metric(validation_scores_file, 'multi_metric[5]')


# ==========================================
# Tests for get_score_map_epoch
# ==========================================

def test_get_score_map_epoch(validation_scores_file):
    """Test get_score_map_epoch with complete scores file."""
    metric_names, num_datapoints, score_map, aggregated_scores = get_score_map_epoch(
        validation_scores_file
    )

    # Check basic structure
    assert isinstance(metric_names, list)
    assert len(metric_names) == 5  # metric1, metric2, multi_metric[0], multi_metric[1], multi_metric[2]
    assert num_datapoints == 5

    # Check score_map shape: (metrics, height, width)
    assert score_map.shape == (5, 3, 3)

    # Check aggregated_scores shape: (metrics,)
    assert aggregated_scores.shape == (5,)

    # Verify metric names are sorted correctly
    expected_metrics = ['metric1', 'metric2', 'multi_metric[0]', 'multi_metric[1]', 'multi_metric[2]']
    assert metric_names == expected_metrics


def test_get_score_map_epoch_invalid_file():
    """Test get_score_map_epoch fails with invalid file format."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'invalid': 'format'}, f)
        invalid_file = f.name

    try:
        with pytest.raises(AssertionError, match="Invalid keys"):
            get_score_map_epoch(invalid_file)
    finally:
        os.unlink(invalid_file)


# ==========================================
# Tests for get_evaluator_score_map
# ==========================================

def test_get_evaluator_score_map(validation_scores_file):
    """Test get_evaluator_score_map with evaluator scores file."""
    metric_names, num_datapoints, score_map, aggregated_scores = get_evaluator_score_map(
        validation_scores_file
    )

    # Check basic structure (same as trainer but different shape)
    assert isinstance(metric_names, list)
    assert len(metric_names) == 5
    assert num_datapoints == 5

    # Check score_map shape: (metrics, height, width) - no epoch dimension
    assert score_map.shape == (5, 3, 3)

    # Check aggregated_scores shape: (metrics,) - no epoch dimension
    assert aggregated_scores.shape == (5,)


# ==========================================
# Tests for get_epoch_dirs
# ==========================================

def test_get_epoch_dirs(trainer_log_structure):
    """Test get_epoch_dirs with valid trainer log structure."""
    log_dir = os.path.dirname(trainer_log_structure[0])
    epoch_dirs = get_epoch_dirs(log_dir)

    assert len(epoch_dirs) == 3  # Created 3 epochs in fixture

    # Check that directories are in correct order
    for i, epoch_dir in enumerate(epoch_dirs):
        assert epoch_dir.endswith(f"epoch_{i}")
        assert os.path.exists(epoch_dir)
        assert os.path.exists(os.path.join(epoch_dir, "validation_scores.json"))


def test_get_epoch_dirs_no_epochs(temp_log_dir):
    """Test get_epoch_dirs fails when no epoch directories exist."""
    with pytest.raises(ValueError, match="No epoch directories with validation scores found"):
        get_epoch_dirs(temp_log_dir)


def test_get_epoch_dirs_missing_scores(temp_log_dir):
    """Test get_epoch_dirs stops at first missing validation_scores.json."""
    # Create epoch_0 and epoch_1 with scores, epoch_2 without scores
    for i in range(2):
        epoch_dir = os.path.join(temp_log_dir, f"epoch_{i}")
        os.makedirs(epoch_dir)
        scores_file = os.path.join(epoch_dir, "validation_scores.json")
        with open(scores_file, 'w') as f:
            json.dump({'aggregated': {}, 'per_datapoint': {}}, f)

    # Create epoch_2 directory but no scores file
    os.makedirs(os.path.join(temp_log_dir, "epoch_2"))

    epoch_dirs = get_epoch_dirs(temp_log_dir)

    # Should only return first 2 epochs
    assert len(epoch_dirs) == 2
    assert epoch_dirs[0].endswith("epoch_0")
    assert epoch_dirs[1].endswith("epoch_1")


# ==========================================
# Tests for compute_per_metric_color_scales
# ==========================================

def test_compute_per_metric_color_scales_trainer_data():
    """Test compute_per_metric_color_scales with trainer data."""
    # Create mock LogDirInfo objects for trainers
    info1 = LogDirInfo(
        num_epochs=2,
        metric_names=['metric1', 'metric2'],
        num_datapoints=4,
        score_map=np.array([  # Shape: (N=2, C=2, H=2, W=2)
            [[0.1, 0.2], [0.3, 0.4]],  # Epoch 0, metric 0
            [[0.5, 0.6], [0.7, 0.8]]   # Epoch 0, metric 1
        ]).reshape(1, 2, 2, 2),  # Reshape to (1, 2, 2, 2) for single epoch
        aggregated_scores=np.array([[0.25, 0.65]]),
        dataset_class='TestDataset',
        dataset_type='semseg',
        dataset_cfg={},
        dataloader_cfg={},
        runner_type='trainer'
    )

    # Stack to create 2 epochs
    info1.score_map = np.tile(info1.score_map, (2, 1, 1, 1))  # Shape: (2, 2, 2, 2)
    info1.aggregated_scores = np.tile(info1.aggregated_scores, (2, 1))  # Shape: (2, 2)

    log_dir_infos = {'log1': info1}

    result = compute_per_metric_color_scales(log_dir_infos)

    # Should return array of shape (2, 2) for 2 metrics
    assert result.shape == (2, 2)

    # Check min/max values for each metric
    # Metric 0: values are [0.1, 0.2, 0.3, 0.4] repeated for 2 epochs
    assert result[0, 0] == 0.1  # Min
    assert result[0, 1] == 0.4  # Max

    # Metric 1: values are [0.5, 0.6, 0.7, 0.8] repeated for 2 epochs
    assert result[1, 0] == 0.5  # Min
    assert result[1, 1] == 0.8  # Max


def test_compute_per_metric_color_scales_evaluator_data():
    """Test compute_per_metric_color_scales with evaluator data."""
    # Create mock LogDirInfo object for evaluator
    info1 = LogDirInfo(
        num_epochs=1,
        metric_names=['metric1'],
        num_datapoints=4,
        score_map=np.array([[0.2, 0.4], [0.6, 0.8]]).reshape(1, 2, 2),  # Shape: (C=1, H=2, W=2)
        aggregated_scores=np.array([0.5]),
        dataset_class='TestDataset',
        dataset_type='semseg',
        dataset_cfg={},
        dataloader_cfg={},
        runner_type='evaluator'
    )

    log_dir_infos = {'log1': info1}

    result = compute_per_metric_color_scales(log_dir_infos)

    # Should return array of shape (1, 2) for 1 metric
    assert result.shape == (1, 2)

    # Check min/max values
    # Metric 0: values are [0.2, 0.4, 0.6, 0.8]
    assert result[0, 0] == 0.2  # Min
    assert result[0, 1] == 0.8  # Max


def test_compute_per_metric_color_scales_with_nans():
    """Test compute_per_metric_color_scales handles NaN values correctly."""
    info1 = LogDirInfo(
        num_epochs=1,
        metric_names=['metric1'],
        num_datapoints=4,
        score_map=np.array([[0.1, np.nan], [0.5, 0.9]]).reshape(1, 2, 2),
        aggregated_scores=np.array([0.5]),
        dataset_class='TestDataset',
        dataset_type='semseg',
        dataset_cfg={},
        dataloader_cfg={},
        runner_type='evaluator'
    )

    log_dir_infos = {'log1': info1}

    result = compute_per_metric_color_scales(log_dir_infos)

    # Should ignore NaN values in min/max calculation
    # Valid values: [0.1, 0.5, 0.9]
    assert result[0, 0] == 0.1  # Min
    assert result[0, 1] == 0.9  # Max


def test_compute_per_metric_color_scales_empty_data():
    """Test compute_per_metric_color_scales with empty data."""
    result = compute_per_metric_color_scales({})

    # Should return default range
    assert np.array_equal(result, np.array([[0.0, 1.0]]))


def test_compute_per_metric_color_scales_all_nans():
    """Test compute_per_metric_color_scales when all values are NaN."""
    info1 = LogDirInfo(
        num_epochs=1,
        metric_names=['metric1'],
        num_datapoints=4,
        score_map=np.full((1, 2, 2), np.nan),
        aggregated_scores=np.array([0.5]),
        dataset_class='TestDataset',
        dataset_type='semseg',
        dataset_cfg={},
        dataloader_cfg={},
        runner_type='evaluator'
    )

    log_dir_infos = {'log1': info1}

    result = compute_per_metric_color_scales(log_dir_infos)

    # Should return default range when no valid scores
    assert np.array_equal(result, np.array([[0.0, 1.0]]))
