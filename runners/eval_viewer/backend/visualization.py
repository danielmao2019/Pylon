from typing import List, Tuple, Dict
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
    Creates a heatmap figure for a score map.

    Args:
        score_map: 2D numpy array of scores
        title: Title for the figure

    Returns:
        fig: Plotly figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=score_map,
        colorscale='Viridis',
        showscale=True,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
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


def create_aggregated_heatmap(score_maps: List[np.ndarray], title: str) -> go.Figure:
    """
    Creates a Plotly figure for an aggregated heatmap showing common failure cases.

    Args:
        score_maps: List of 2D numpy arrays containing scores from different runs
        title: Title for the figure

    Returns:
        figure: Plotly figure object
    """
    # Flatten all score maps to find the global score distribution
    all_scores = np.concatenate([score_map.flatten() for score_map in score_maps])
    all_scores = all_scores[~np.isnan(all_scores)]  # Remove NaN values
    
    # Use the 25th percentile as the failure threshold
    # This means scores below this threshold are considered "failures"
    failure_threshold = np.percentile(all_scores, 25)
    
    # Convert all score maps to binary masks (1 for failures, 0 for non-failures)
    binary_maps = [score_map < failure_threshold for score_map in score_maps]
    
    # Sum the binary maps to get the number of runs that failed at each position
    aggregated = np.sum(binary_maps, axis=0)
    
    # Normalize to [0, 1] range
    normalized = aggregated / len(score_maps)
    fig = go.Figure(data=go.Heatmap(
        z=normalized,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(
            title="Failure Rate",
            ticktext=['0%', '25%', '50%', '75%', '100%'],
            tickvals=[0, 0.25, 0.5, 0.75, 1]
        ),
        hoverinfo='x+y+z',
    ))

    fig.update_layout(
        title=f"{title} (Failure threshold: {failure_threshold:.3f})",
        xaxis_title="Column",
        yaxis_title="Row",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig


def create_aggregated_scores_plot(epoch_scores: List[Dict[str, float]], log_dirs: List[str], metric: str) -> go.Figure:
    """
    Creates a line plot showing aggregated scores over epochs for each run.

    Args:
        epoch_scores: List of dictionaries containing aggregated scores for each epoch
        log_dirs: List of log directory paths
        metric: Name of the metric to plot

    Returns:
        fig: Plotly figure object
    """
    fig = go.Figure()

    for i, (scores, log_dir) in enumerate(zip(epoch_scores, log_dirs)):
        run_name = log_dir.split('/')[-1]
        
        # Extract scores for the selected metric
        if '[' in metric:
            base_metric, idx_str = metric.split('[')
            idx = int(idx_str.rstrip(']'))
            y_values = [scores['aggregated'][base_metric][idx] for scores in epoch_scores[i]]
        else:
            y_values = [scores['aggregated'][metric] for scores in epoch_scores[i]]
        
        x_values = list(range(len(y_values)))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            name=run_name,
            mode='lines+markers'
        ))

    fig.update_layout(
        title=f"Aggregated {metric} Over Time",
        xaxis_title="Epoch",
        yaxis_title="Score",
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    
    return fig
