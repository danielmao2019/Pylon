"""Real integration tests for eval_viewer using actual trainer and evaluator result files."""

import pytest
import os
import tempfile
import shutil
from pathlib import Path

from runners.viewers.eval_viewer.backend.initialization import (
    initialize_log_dirs,
    extract_log_dir_info,
    LogDirInfo
)


@pytest.fixture(scope="module")
def real_trainer_log_dir():
    """Path to real trainer log directory. Generates test data if not found."""
    import subprocess

    trainer_dir = "./logs/tests/runners/eval_viewer/trainer_integration_test_run_0"
    if not os.path.exists(trainer_dir):
        # Generate test data by running the trainer config
        result = subprocess.run([
            "python", "main.py",
            "--config-filepath", "configs/tests/runners/eval_viewer/trainer_integration_test_run_0.py"
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"Failed to generate trainer test data: {result.stderr}"
        assert os.path.exists(trainer_dir), f"Trainer log directory not created: {trainer_dir}"

    return trainer_dir


@pytest.fixture(scope="module")
def real_evaluator_log_dir():
    """Path to real evaluator log directory. Generates test data if not found."""
    import subprocess

    evaluator_dir = "./logs/tests/runners/eval_viewer/evaluator_integration_test_run_0"
    if not os.path.exists(evaluator_dir):
        # Generate test data by running the evaluator config
        result = subprocess.run([
            "python", "main.py",
            "--config-filepath", "configs/tests/runners/eval_viewer/evaluator_integration_test_run_0.py"
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"Failed to generate evaluator test data: {result.stderr}"
        assert os.path.exists(evaluator_dir), f"Evaluator log directory not created: {evaluator_dir}"

    return evaluator_dir


def test_real_trainer_workflow_integration(real_trainer_log_dir):
    """Test complete trainer workflow using real result files."""
    log_dir = real_trainer_log_dir

    # Test complete workflow with real data using actual config files
    log_dir_info = extract_log_dir_info(log_dir, force_reload=True)

    # Verify the integrated pipeline worked correctly with real data
    assert isinstance(log_dir_info, LogDirInfo)
    assert log_dir_info.runner_type == 'trainer'
    assert log_dir_info.num_epochs == 3  # From our trainer config

    # Verify metrics were extracted from real files
    assert log_dir_info.metric_names == ['score']  # Single metric from PyTorchMetricWrapper
    assert log_dir_info.num_datapoints == 16  # From our dataset config

    # Verify score processing with real data
    assert log_dir_info.score_map.shape == (3, 1, 4, 4)  # 3 epochs, 1 metric, 4x4 grid
    assert log_dir_info.aggregated_scores.shape == (3, 1)

    # Verify data quality - scores should be positive and realistic
    assert (log_dir_info.aggregated_scores >= 0).all()
    assert (log_dir_info.aggregated_scores < 1.0).all()  # MSE scores should be reasonable

    # Verify per-datapoint scores are properly gridded
    # First 16 positions should have real values, rest should be NaN
    import numpy as np
    score_map_flat = log_dir_info.score_map[0, 0].flatten()  # First epoch, first metric
    assert not any(np.isnan(score_map_flat[:16]))  # First 16 should be valid
    assert all(np.isnan(score_map_flat[16:]))  # Rest should be NaN

    # Verify dataset and config info was extracted correctly
    assert log_dir_info.dataset_class == 'BaseRandomDataset'
    assert log_dir_info.dataset_type == 'general'  # Default for BaseRandomDataset


def test_real_evaluator_workflow_integration(real_evaluator_log_dir):
    """Test complete evaluator workflow using real result files."""
    log_dir = real_evaluator_log_dir

    # Test complete workflow with real evaluator data using actual config files
    log_dir_info = extract_log_dir_info(log_dir, force_reload=True)

    # Verify evaluator-specific pipeline behavior with real data
    assert isinstance(log_dir_info, LogDirInfo)
    assert log_dir_info.runner_type == 'evaluator'
    assert log_dir_info.num_epochs == 1  # Evaluators have single evaluation

    # Same metrics processing as trainer but with real evaluator data
    assert log_dir_info.metric_names == ['score']
    assert log_dir_info.num_datapoints == 16

    # Verify evaluator-specific data shapes (no epoch dimension)
    assert log_dir_info.score_map.shape == (1, 4, 4)  # 1 metric, 4x4 grid (no epoch dim)
    assert log_dir_info.aggregated_scores.shape == (1,)  # 1 metric (no epoch dim)

    # Verify data quality with real evaluation scores
    assert (log_dir_info.aggregated_scores >= 0).all()
    assert (log_dir_info.aggregated_scores < 1.0).all()


def test_mixed_real_runs_integration(real_trainer_log_dir, real_evaluator_log_dir):
    """Test integration with mixed real trainer and evaluator runs."""
    trainer_log_dir = real_trainer_log_dir
    evaluator_log_dir = real_evaluator_log_dir

    # Test complete initialization workflow with real mixed runs using actual config files
    log_dirs = [trainer_log_dir, evaluator_log_dir]
    result = initialize_log_dirs(log_dirs, force_reload=True)
    max_epochs, metric_names, num_datapoints, dataset_cfg, dataset_type, log_dir_infos, color_scales = result

    # Verify integrated initialization worked with real data
    assert len(log_dir_infos) == 2
    assert "trainer_integration_test" in log_dir_infos
    assert "evaluator_integration_test" in log_dir_infos

    # Verify data consistency across different real run types
    trainer_info = log_dir_infos["trainer_integration_test"]
    evaluator_info = log_dir_infos["evaluator_integration_test"]

    assert trainer_info.runner_type == 'trainer'
    assert evaluator_info.runner_type == 'evaluator'

    # Integration should ensure consistent metrics across real runs
    assert trainer_info.metric_names == evaluator_info.metric_names
    assert trainer_info.num_datapoints == evaluator_info.num_datapoints

    # Verify color scale computation with real data
    assert color_scales.shape == (1, 2)  # 1 metric, min/max for each

    # Verify that color scales incorporate real data from both runs
    min_score, max_score = color_scales[0]
    assert min_score <= max_score
    # Should be reasonable ranges based on real MSE scores
    assert 0.0 <= min_score <= 1.0
    assert 0.0 <= max_score <= 1.0

    # Verify global consistency values
    assert max_epochs == 3  # From trainer
    assert metric_names == ['score']
    assert num_datapoints == 16
    assert dataset_type == 'general'


def test_real_cache_integration_workflow(real_trainer_log_dir):
    """Test that caching works correctly with real result files."""
    log_dir = real_trainer_log_dir

    # First extraction should process real files and create cache using actual config files
    log_dir_info1 = extract_log_dir_info(log_dir, force_reload=True)

    # Second extraction should use cache but return same real results
    log_dir_info2 = extract_log_dir_info(log_dir, force_reload=False)

    # Verify that caching preserves real pipeline results
    assert log_dir_info1.num_epochs == log_dir_info2.num_epochs
    assert log_dir_info1.metric_names == log_dir_info2.metric_names
    assert log_dir_info1.runner_type == log_dir_info2.runner_type

    # Verify that complex real data structures are cached correctly
    import numpy as np
    np.testing.assert_array_equal(log_dir_info1.score_map, log_dir_info2.score_map)
    np.testing.assert_array_equal(log_dir_info1.aggregated_scores, log_dir_info2.aggregated_scores)


def test_real_error_propagation_integration(real_trainer_log_dir):
    """Test error handling with real log structure but corrupted data."""
    import json

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create log directory structure that mirrors the expected config path
        logs_dir = os.path.join(temp_dir, "logs", "tests", "runners", "eval_viewer")
        os.makedirs(logs_dir, exist_ok=True)
        temp_log_dir = os.path.join(logs_dir, "corrupted_trainer_run_0")
        shutil.copytree(real_trainer_log_dir, temp_log_dir)

        # Create corresponding config directory structure and copy config
        config_dir = os.path.join(temp_dir, "configs", "tests", "runners", "eval_viewer")
        os.makedirs(config_dir, exist_ok=True)
        shutil.copy2("configs/tests/runners/eval_viewer/trainer_integration_test_run_0.py",
                    os.path.join(config_dir, "corrupted_trainer_run_0.py"))

        # Corrupt one of the real score files
        corrupt_file = os.path.join(temp_log_dir, "epoch_1", "validation_scores.json")
        with open(corrupt_file, 'w') as f:
            f.write("invalid json {")

        # Temporarily change working directory for this test
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            # Should fail fast and loud with real error (following CLAUDE.md philosophy)
            with pytest.raises(json.JSONDecodeError):
                extract_log_dir_info("./logs/tests/runners/eval_viewer/corrupted_trainer_run_0", force_reload=True)
        finally:
            os.chdir(original_cwd)


def test_real_callback_data_flow_integration(real_trainer_log_dir, real_evaluator_log_dir):
    """Test that real data flows correctly from backend to callback functions."""
    from runners.viewers.eval_viewer.callbacks.update_plots import create_aggregated_scores_plot

    # Extract real data from both runs using actual config files
    trainer_info = extract_log_dir_info(real_trainer_log_dir, force_reload=True)
    evaluator_info = extract_log_dir_info(real_evaluator_log_dir, force_reload=True)

    # Create epoch scores from real data
    trainer_epochs = trainer_info.aggregated_scores[:, 0]  # All epochs for 'score' metric
    evaluator_single = evaluator_info.aggregated_scores  # Single evaluation result

    epoch_scores = [trainer_epochs, evaluator_single]
    log_dirs = [real_trainer_log_dir, real_evaluator_log_dir]
    metric_name = 'score'

    # Test that callback can handle real mixed data from backend integration
    fig = create_aggregated_scores_plot(
        epoch_scores=epoch_scores,
        log_dirs=log_dirs,
        metric_name=metric_name
    )

    # Verify integration between real backend data and callback visualization
    assert len(fig.data) == 2  # Both trainer and evaluator data

    # Verify trainer data shows real progression over 3 epochs
    trainer_trace = fig.data[0]
    assert len(trainer_trace.y) == 3  # Three epochs from real trainer

    # Verify evaluator data shows single real point
    evaluator_trace = fig.data[1]
    assert len(evaluator_trace.y) == 1  # Single evaluation from real evaluator

    # Verify data values are from real results (not mocked)
    import numpy as np
    np.testing.assert_array_equal(trainer_trace.y, trainer_epochs)
    np.testing.assert_array_equal(evaluator_trace.y, evaluator_single)


def test_performance_with_real_data_integration(real_trainer_log_dir):
    """Test performance characteristics with real data processing."""
    import time
    import numpy as np

    # Measure performance of real data processing using actual config files
    start_time = time.time()
    log_dir_info = extract_log_dir_info(real_trainer_log_dir, force_reload=True)
    processing_time = time.time() - start_time

    # With our small test dataset, processing should be very fast
    assert processing_time < 5.0  # Should complete within 5 seconds

    # Verify data was processed correctly despite speed requirements
    assert log_dir_info.score_map.shape == (3, 1, 4, 4)
    assert not any(np.isnan(log_dir_info.aggregated_scores))

    # Test cached access is even faster
    start_time = time.time()
    cached_info = extract_log_dir_info(real_trainer_log_dir, force_reload=False)
    cached_time = time.time() - start_time

    # Cached access should be significantly faster
    assert cached_time < processing_time / 2

    # Results should be identical
    np.testing.assert_array_equal(log_dir_info.score_map, cached_info.score_map)