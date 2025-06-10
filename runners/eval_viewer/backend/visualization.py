from typing import List
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


def get_color_for_score(score: float, min_score: float, max_score: float) -> str:
    """Convert a score to a color using a red-yellow-green colormap."""
    if np.isnan(score):
        return '#808080'  # Gray for NaN values

    # Normalize score to [0, 1]
    normalized = (score - min_score) / (max_score - min_score)

    # Create color gradient from red (0) to yellow (0.5) to green (1)
    if normalized < 0.5:
        # Red to Yellow
        r = 1.0
        g = normalized * 2
        b = 0.0
    else:
        # Yellow to Green
        r = 2 * (1 - normalized)
        g = 1.0
        b = 0.0

    return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
