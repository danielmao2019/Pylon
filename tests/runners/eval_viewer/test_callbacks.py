"""Tests for callback integration and plot generation functions."""

import pytest
import numpy as np
from runners.eval_viewer.callbacks.update_plots import (
    create_aggregated_scores_plot,
    create_grid_and_colorbar
)


def test_plot_generation():
    """Test that plot generation functions work with realistic data."""
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
    
    # Verify figure structure
    assert hasattr(fig, 'data')
    assert len(fig.data) == 2  # Two runs
    assert fig.layout.title.text == f"Aggregated {metric_name} Over Time"
    assert fig.layout.xaxis.title.text == "Epoch"
    assert fig.layout.yaxis.title.text == "Score"


def test_grid_and_colorbar_generation():
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
    
    # Verify results
    assert result_run_idx == run_idx
    assert button_grid is not None
    assert color_bar is not None