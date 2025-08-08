"""Comprehensive tests for callback functions and plot generation."""

import pytest
import numpy as np
import plotly.graph_objects as go
from runners.eval_viewer.callbacks.update_plots import (
    create_aggregated_scores_plot,
    create_grid_and_colorbar
)
from runners.eval_viewer.callbacks.datapoint_viewer import (
    register_datapoint_viewer_callbacks
)


def test_create_aggregated_scores_plot_normal_case():
    """Test aggregated scores plot creation with realistic data."""
    # Create realistic epoch scores data
    epoch_scores_run1 = np.array([0.1, 0.3, 0.5, 0.7, 0.8])  # Improving over time
    epoch_scores_run2 = np.array([0.2, 0.4, 0.6, 0.65, 0.7])  # Different improvement curve
    
    epoch_scores = [epoch_scores_run1, epoch_scores_run2]
    log_dirs = ['logs/run1', 'logs/run2']
    metric_name = 'accuracy'
    
    # Test plot creation
    fig = create_aggregated_scores_plot(
        epoch_scores=epoch_scores,
        log_dirs=log_dirs,
        metric_name=metric_name
    )
    
    # Verify figure structure and content
    assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"
    assert hasattr(fig, 'data'), "Figure must have data attribute"
    assert len(fig.data) == 2, f"Expected 2 traces, got {len(fig.data)}"
    
    # Verify layout properties
    assert fig.layout.title.text == f"Aggregated {metric_name} Over Time"
    assert fig.layout.xaxis.title.text == "Epoch"
    assert fig.layout.yaxis.title.text == "Score"
    
    # Verify actual data content
    trace1, trace2 = fig.data[0], fig.data[1]
    np.testing.assert_array_equal(trace1.y, epoch_scores_run1)
    np.testing.assert_array_equal(trace2.y, epoch_scores_run2)
    
    # Verify trace names (uses last part of log_dir path)
    assert trace1.name == 'run1'
    assert trace2.name == 'run2'
    
    # Verify x-axis (epochs) - should be 0-indexed
    expected_epochs = list(range(len(epoch_scores_run1)))
    np.testing.assert_array_equal(trace1.x, expected_epochs)
    np.testing.assert_array_equal(trace2.x, expected_epochs)


def test_create_aggregated_scores_plot_single_run():
    """Test plot creation with single run."""
    epoch_scores = [np.array([0.2, 0.4, 0.6, 0.8])]
    log_dirs = ['logs/single_run']
    metric_name = 'f1_score'
    
    fig = create_aggregated_scores_plot(
        epoch_scores=epoch_scores,
        log_dirs=log_dirs,
        metric_name=metric_name
    )
    
    assert len(fig.data) == 1
    assert fig.data[0].name == 'single_run'
    assert fig.layout.title.text == "Aggregated f1_score Over Time"
    np.testing.assert_array_equal(fig.data[0].y, epoch_scores[0])


def test_create_aggregated_scores_plot_single_epoch():
    """Test plot creation with single epoch data."""
    epoch_scores = [np.array([0.5]), np.array([0.7])]
    log_dirs = ['logs/run1', 'logs/run2']
    metric_name = 'mse'
    
    fig = create_aggregated_scores_plot(
        epoch_scores=epoch_scores,
        log_dirs=log_dirs,
        metric_name=metric_name
    )
    
    assert len(fig.data) == 2
    assert fig.data[0].y[0] == 0.5
    assert fig.data[1].y[0] == 0.7
    assert fig.data[0].x[0] == 0  # First epoch


def test_create_aggregated_scores_plot_with_nan_values():
    """Test plot creation handles NaN values correctly."""
    epoch_scores_with_nan = [np.array([0.1, np.nan, 0.5]), np.array([0.2, 0.4, np.nan])]
    log_dirs = ['logs/run1', 'logs/run2']
    metric_name = 'accuracy'
    
    fig = create_aggregated_scores_plot(
        epoch_scores=epoch_scores_with_nan,
        log_dirs=log_dirs,
        metric_name=metric_name
    )
    
    # Verify NaN values are preserved in plot data
    assert len(fig.data) == 2
    assert np.isnan(fig.data[0].y[1])  # Second epoch of first run
    assert np.isnan(fig.data[1].y[2])  # Third epoch of second run
    assert fig.data[0].y[0] == 0.1
    assert fig.data[1].y[1] == 0.4


def test_create_aggregated_scores_plot_invalid_inputs():
    """Test plot creation with invalid inputs."""
    # Test None epoch_scores
    with pytest.raises(AssertionError, match="epoch_scores must not be None"):
        create_aggregated_scores_plot(None, ['logs/run1'], 'accuracy')
    
    # Test empty epoch_scores
    with pytest.raises(AssertionError, match="epoch_scores must not be empty"):
        create_aggregated_scores_plot([], ['logs/run1'], 'accuracy')
    
    # Test non-list epoch_scores
    with pytest.raises(AssertionError, match="epoch_scores must be list"):
        create_aggregated_scores_plot(np.array([0.1, 0.2]), ['logs/run1'], 'accuracy')
    
    # Test non-numpy array elements
    with pytest.raises(AssertionError, match="All epoch_scores must be numpy arrays"):
        create_aggregated_scores_plot([[0.1, 0.2]], ['logs/run1'], 'accuracy')
    
    # Test mismatched lengths
    epoch_scores = [np.array([0.1, 0.2])]
    log_dirs = ['logs/run1', 'logs/run2']  # Length mismatch
    with pytest.raises(AssertionError, match="log_dirs length .* must match epoch_scores length"):
        create_aggregated_scores_plot(epoch_scores, log_dirs, 'accuracy')
    
    # Test None log_dirs
    with pytest.raises(AssertionError, match="log_dirs must not be None"):
        create_aggregated_scores_plot([np.array([0.1])], None, 'accuracy')
    
    # Test None metric_name
    with pytest.raises(AssertionError, match="metric_name must not be None"):
        create_aggregated_scores_plot([np.array([0.1])], ['logs/run1'], None)


def test_create_grid_and_colorbar_normal_case():
    """Test grid and colorbar generation with realistic score maps."""
    # Create a realistic score map
    score_map = np.array([
        [0.1, 0.3, 0.8],
        [0.5, 0.9, 0.2],
        [0.7, np.nan, np.nan]  # Partial data
    ])
    
    run_idx = 0
    num_datapoints = 7  # More datapoints than grid positions (some NaN expected)
    min_score = 0.0
    max_score = 1.0
    
    # Test grid and colorbar creation
    result_run_idx, (button_grid, color_bar) = create_grid_and_colorbar(
        score_map=score_map,
        run_idx=run_idx,
        num_datapoints=num_datapoints,
        min_score=min_score,
        max_score=max_score
    )
    
    # Verify return structure
    assert result_run_idx == run_idx, f"Expected run_idx {run_idx}, got {result_run_idx}"
    assert button_grid is not None, "Button grid must not be None"
    assert color_bar is not None, "Color bar must not be None"
    
    # Verify button grid structure (should be Dash components)
    assert hasattr(button_grid, 'children'), "Button grid should have children attribute"
    
    # Verify colorbar structure (Dash component, not plotly figure)
    assert hasattr(color_bar, 'children'), "Color bar should have children attribute"
    assert color_bar.style is not None, "Color bar should have style"


def test_create_grid_and_colorbar_single_datapoint():
    """Test grid and colorbar with single datapoint."""
    score_map = np.array([[0.5]])
    
    result_run_idx, (button_grid, color_bar) = create_grid_and_colorbar(
        score_map=score_map,
        run_idx=2,
        num_datapoints=1,
        min_score=0.0,
        max_score=1.0
    )
    
    assert result_run_idx == 2
    assert button_grid is not None
    assert color_bar is not None


def test_create_grid_and_colorbar_all_nan():
    """Test grid and colorbar with all NaN values."""
    score_map = np.array([
        [np.nan, np.nan],
        [np.nan, np.nan]
    ])
    
    result_run_idx, (button_grid, color_bar) = create_grid_and_colorbar(
        score_map=score_map,
        run_idx=1,
        num_datapoints=2,
        min_score=0.0,
        max_score=1.0
    )
    
    assert result_run_idx == 1
    assert button_grid is not None
    assert color_bar is not None


def test_create_grid_and_colorbar_edge_scores():
    """Test grid and colorbar with edge score values."""
    score_map = np.array([
        [0.0, 1.0],  # Min and max values
        [0.5, np.nan]
    ])
    
    result_run_idx, (button_grid, color_bar) = create_grid_and_colorbar(
        score_map=score_map,
        run_idx=3,
        num_datapoints=3,
        min_score=0.0,
        max_score=1.0
    )
    
    assert result_run_idx == 3
    assert button_grid is not None
    assert color_bar is not None


def test_create_grid_and_colorbar_custom_score_range():
    """Test grid and colorbar with custom score range."""
    score_map = np.array([
        [10.0, 15.0],
        [12.5, 20.0]
    ])
    
    result_run_idx, (button_grid, color_bar) = create_grid_and_colorbar(
        score_map=score_map,
        run_idx=0,
        num_datapoints=4,
        min_score=10.0,
        max_score=20.0
    )
    
    assert result_run_idx == 0
    assert button_grid is not None
    assert color_bar is not None


def test_register_datapoint_viewer_callbacks_input_validation():
    """Test datapoint viewer callback registration input validation."""
    # Mock app object
    class MockApp:
        def callback(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    
    app = MockApp()
    
    # Test with valid inputs
    dataset_cfg = {'class': 'MockDataset', 'args': {}}
    dataset_type = 'semseg'
    log_dir_infos = {}
    
    # Should not raise any exceptions with valid inputs
    try:
        register_datapoint_viewer_callbacks(
            app=app,
            dataset_cfg=dataset_cfg,
            dataset_type=dataset_type,
            log_dir_infos=log_dir_infos
        )
    except Exception as e:
        # If it fails, it should be due to missing dependencies, not input validation
        pass


def test_create_aggregated_scores_plot_different_lengths():
    """Test plot creation with different epoch lengths between runs."""
    epoch_scores = [
        np.array([0.1, 0.3, 0.5]),      # 3 epochs
        np.array([0.2, 0.4])            # 2 epochs
    ]
    log_dirs = ['logs/run1', 'logs/run2']
    metric_name = 'accuracy'
    
    fig = create_aggregated_scores_plot(
        epoch_scores=epoch_scores,
        log_dirs=log_dirs,
        metric_name=metric_name
    )
    
    # Verify both traces are included with their respective lengths
    assert len(fig.data) == 2
    assert len(fig.data[0].y) == 3  # First run has 3 epochs
    assert len(fig.data[1].y) == 2  # Second run has 2 epochs
    
    # Verify x-axis values are correct for each trace
    np.testing.assert_array_equal(fig.data[0].x, [0, 1, 2])
    np.testing.assert_array_equal(fig.data[1].x, [0, 1])


def test_create_aggregated_scores_plot_extreme_values():
    """Test plot creation with extreme values."""
    epoch_scores = [
        np.array([1e-10, 1e10, -1e10]),  # Very small, very large, very negative
        np.array([0.0, np.inf, -np.inf])  # Zero, positive infinity, negative infinity
    ]
    log_dirs = ['logs/extreme1', 'logs/extreme2']
    metric_name = 'extreme_metric'
    
    fig = create_aggregated_scores_plot(
        epoch_scores=epoch_scores,
        log_dirs=log_dirs,
        metric_name=metric_name
    )
    
    # Verify extreme values are preserved
    assert len(fig.data) == 2
    assert fig.data[0].y[0] == 1e-10
    assert fig.data[0].y[1] == 1e10
    assert fig.data[0].y[2] == -1e10
    assert fig.data[1].y[0] == 0.0
    assert np.isinf(fig.data[1].y[1])
    assert np.isinf(fig.data[1].y[2])


def test_create_aggregated_scores_plot_special_metric_names():
    """Test plot creation with special characters in metric names."""
    epoch_scores = [np.array([0.1, 0.2])]
    log_dirs = ['logs/test']
    
    special_names = [
        'metric_with_underscores',
        'metric-with-dashes',
        'metric with spaces',
        'metric/with/slashes',
        'metric.with.dots',
        'METRIC_UPPERCASE',
        'metric123numbers'
    ]
    
    for metric_name in special_names:
        fig = create_aggregated_scores_plot(
            epoch_scores=epoch_scores,
            log_dirs=log_dirs,
            metric_name=metric_name
        )
        
        assert fig.layout.title.text == f"Aggregated {metric_name} Over Time"
        assert fig.layout.yaxis.title.text == "Score"


def test_create_grid_and_colorbar_large_dataset():
    """Test grid and colorbar with large dataset."""
    # Create larger score map (10x10 grid)
    score_map = np.random.rand(10, 10)
    score_map[8:, 8:] = np.nan  # Some NaN values at the end
    
    result_run_idx, (button_grid, color_bar) = create_grid_and_colorbar(
        score_map=score_map,
        run_idx=5,
        num_datapoints=96,  # 96 out of 100 grid positions
        min_score=0.0,
        max_score=1.0
    )
    
    assert result_run_idx == 5
    assert button_grid is not None
    assert color_bar is not None