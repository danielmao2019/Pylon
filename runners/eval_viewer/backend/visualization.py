from typing import List, Tuple
import numpy as np
import plotly.graph_objects as go


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


def create_score_map_figure(score_map: np.ndarray, title: str) -> go.Figure:
    """
    Creates a Plotly figure for a score map.

    Args:
        score_map: 2D numpy array containing scores
        title: Title for the figure

    Returns:
        figure: Plotly figure object
    """
    assert score_map.ndim == 2, f"Score map must be 2D, got {score_map.ndim}D"

    color_scale = get_color_scale()

    fig = go.Figure(data=go.Heatmap(
        z=score_map,
        colorscale=color_scale,
        showscale=True,
        colorbar=dict(title="Score"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Column",
        yaxis_title="Row",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig


def get_color_scale() -> List[Tuple[float, str]]:
    """
    Returns the color scale for score visualization.

    Returns:
        color_scale: List of (value, color) tuples for the color scale
    """
    return [
        [0, 'rgb(255, 255, 0)'],    # Yellow for low scores
        [0.5, 'rgb(255, 255, 255)'], # White for middle scores
        [1, 'rgb(0, 0, 255)'],      # Blue for high scores
    ]
