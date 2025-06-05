from typing import List
import numpy as np
import plotly.graph_objects as go


def create_aggregated_scores_plot(epoch_scores: List[np.ndarray], log_dirs: List[str], metric_name: str) -> go.Figure:
    """
    Creates a line plot showing aggregated scores over epochs for each run.

    Args:
        epoch_scores: List of dictionaries containing aggregated scores for each epoch
        log_dirs: List of log directory paths
        metric_name: Name of the metric to plot

    Returns:
        fig: Plotly figure object
    """
    assert isinstance(epoch_scores, list)
    assert all(isinstance(scores, np.ndarray) for scores in epoch_scores)

    fig = go.Figure()

    for scores, log_dir in zip(epoch_scores, log_dirs):
        fig.add_trace(go.Scatter(
            x=list(range(len(scores))),
            y=scores,
            name=log_dir.split('/')[-1],
            mode='lines+markers'
        ))

    fig.update_layout(
        title=f"Aggregated {metric_name} Over Time",
        xaxis_title="Epoch",
        yaxis_title="Score",
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig


def create_score_map(scores: List[float]) -> np.ndarray:
    """
    Creates a square matrix from a list of scores.

    Args:
        scores: List of float scores

    Returns:
        score_map: 2D numpy array of shape (H, W) where H*W >= len(scores)
    """
    n = len(scores)
    side_length = int(np.ceil(np.sqrt(n)))

    # Create array of NaN's
    score_map = np.full((side_length, side_length), np.nan)

    # Reshape scores into a square matrix, filling with NaN's
    score_map.flat[:n] = scores

    return score_map


def create_overlaid_score_map(score_maps: List[np.ndarray]) -> np.ndarray:
    """
    Returns the normalized overlaid score map (failure rate map) as a 2D numpy array.
    Args:
        score_maps: List of 2D numpy arrays containing scores from different runs
        title: (Unused, kept for compatibility)
    Returns:
        normalized: 2D numpy array of normalized failure rates
    """
    all_scores = np.concatenate([score_map.flatten() for score_map in score_maps])
    all_scores = all_scores[~np.isnan(all_scores)]
    failure_threshold = np.percentile(all_scores, 25)
    binary_maps = [score_map < failure_threshold for score_map in score_maps]
    aggregated = np.sum(binary_maps, axis=0)
    normalized = aggregated / len(score_maps)
    return normalized
