"""Tests for data validation and consistency checking."""

import pytest
import numpy as np
from runners.viewers.eval_viewer.backend.initialization import (
    LogDirInfo,
    _validate_log_dir_consistency,
    compute_per_metric_color_scales
)


def test_data_consistency_validation():
    """Test that data consistency validation works properly."""
    # Create LogDirInfo objects with inconsistent data
    info1 = LogDirInfo(
        num_epochs=2,
        metric_names=['metric1', 'metric2'],
        num_datapoints=100,
        score_map=np.random.random((2, 2, 10, 10)),
        aggregated_scores=np.random.random((2, 2)),
        dataset_class='DatasetA',
        dataset_type='semseg',
        dataset_cfg={},
        dataloader_cfg={},
        runner_type='trainer'
    )

    info2 = LogDirInfo(
        num_epochs=1,
        metric_names=['metric1', 'metric3'],  # Different metrics
        num_datapoints=100,
        score_map=np.random.random((2, 10, 10)),
        aggregated_scores=np.random.random((2,)),
        dataset_class='DatasetA',
        dataset_type='semseg',
        dataset_cfg={},
        dataloader_cfg={},
        runner_type='evaluator'
    )

    log_dir_infos = {'log1': info1, 'log2': info2}

    # Should raise assertion error due to inconsistent metrics
    with pytest.raises(AssertionError, match="Inconsistent metric names"):
        _validate_log_dir_consistency(log_dir_infos)


def test_color_scale_computation_integration():
    """Test color scale computation with realistic LogDirInfo objects."""
    # Create realistic LogDirInfo objects
    trainer_info = LogDirInfo(
        num_epochs=3,
        metric_names=['accuracy', 'loss'],
        num_datapoints=25,
        score_map=np.random.uniform(0.1, 0.9, (3, 2, 5, 5)),  # (N, C, H, W)
        aggregated_scores=np.random.uniform(0.2, 0.8, (3, 2)),  # (N, C)
        dataset_class='TestDataset',
        dataset_type='semseg',
        dataset_cfg={},
        dataloader_cfg={},
        runner_type='trainer'
    )

    evaluator_info = LogDirInfo(
        num_epochs=1,
        metric_names=['accuracy', 'loss'],
        num_datapoints=25,
        score_map=np.random.uniform(0.1, 0.9, (2, 5, 5)),  # (C, H, W)
        aggregated_scores=np.random.uniform(0.2, 0.8, (2,)),  # (C,)
        dataset_class='TestDataset',
        dataset_type='semseg',
        dataset_cfg={},
        dataloader_cfg={},
        runner_type='evaluator'
    )

    log_dir_infos = {
        'trainer_run': trainer_info,
        'evaluator_run': evaluator_info
    }

    # Test color scale computation
    color_scales = compute_per_metric_color_scales(log_dir_infos)

    # Verify shape and values
    assert color_scales.shape == (2, 2)  # 2 metrics, [min, max] for each

    # Verify that min <= max for each metric
    for metric_idx in range(2):
        min_score, max_score = color_scales[metric_idx]
        assert min_score <= max_score
        assert 0.0 <= min_score <= 1.0
        assert 0.0 <= max_score <= 1.0


def test_nan_handling_integration():
    """Test NaN handling in integrated pipeline."""
    # Create data with NaN values
    score_map_with_nans = np.array([
        [[0.5, 0.7, np.nan], [0.2, np.nan, 0.8], [0.9, 0.1, 0.6]],  # Metric 1
        [[np.nan, 0.3, 0.4], [0.7, 0.8, np.nan], [0.2, 0.9, 0.5]]   # Metric 2
    ])

    aggregated_with_nans = np.array([0.6, np.nan])

    log_dir_info = LogDirInfo(
        num_epochs=1,
        metric_names=['metric1', 'metric2'],
        num_datapoints=9,
        score_map=score_map_with_nans,
        aggregated_scores=aggregated_with_nans,
        dataset_class='TestDataset',
        dataset_type='semseg',
        dataset_cfg={},
        dataloader_cfg={},
        runner_type='evaluator'
    )

    log_dir_infos = {'test_run': log_dir_info}

    # Test color scale computation with NaN values
    color_scales = compute_per_metric_color_scales(log_dir_infos)

    # Should handle NaN values gracefully
    assert color_scales.shape == (2, 2)

    # First metric should have valid range
    assert not np.isnan(color_scales[0]).any()

    # Second metric might have default range due to NaN aggregated score
    # but should still have valid min/max values
    assert not np.isnan(color_scales[1]).any()