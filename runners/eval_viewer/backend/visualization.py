from typing import List, Union
import numpy as np
import plotly.graph_objects as go


def create_score_map_grid(scores: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Creates a square matrix from scores (list or numpy array).

    Args:
        scores: List of float scores or 1D numpy array of scores

    Returns:
        score_map: 2D numpy array of shape (H, W) where H*W >= len(scores)
    """
    # Input validation following CLAUDE.md fail-fast patterns
    assert scores is not None, "scores must not be None"
    assert isinstance(scores, (list, np.ndarray)), f"scores must be list or numpy array, got {type(scores)}"
    
    if isinstance(scores, list):
        assert len(scores) > 0, f"scores must not be empty"
        assert all(isinstance(score, (int, float)) for score in scores), f"All scores must be numeric"
        scores_array = np.array(scores)
    else:
        assert scores.ndim == 1, f"scores must be 1D array, got shape {scores.shape}"
        assert len(scores) > 0, f"scores must not be empty"
        scores_array = scores
    
    n = len(scores_array)
    side_length = int(np.ceil(np.sqrt(n)))

    # Create array of NaN's
    score_map = np.full((side_length, side_length), np.nan)

    # Reshape scores into a square matrix, filling with NaN's
    score_map.flat[:n] = scores_array

    return score_map




def create_overlaid_score_map(score_maps: List[np.ndarray], percentile: float = 25) -> np.ndarray:
    """
    Returns the normalized overlaid score map (success rate map) as a 2D numpy array.
    Args:
        score_maps: List of 2D numpy arrays containing scores from different runs
        percentile: Percentile threshold for failure (default: 25)
    Returns:
        normalized: 2D numpy array of normalized success rates (1 - failure rates)
    """
    all_scores = np.concatenate([score_map.flatten() for score_map in score_maps])
    all_scores = all_scores[~np.isnan(all_scores)]
    failure_threshold = np.percentile(all_scores, percentile)
    binary_maps = [score_map < failure_threshold for score_map in score_maps]
    aggregated = np.sum(binary_maps, axis=0)
    failure_rates = aggregated / len(score_maps)
    # Return success rates (1 - failure_rates) so higher values are better
    success_rates = 1 - failure_rates
    return success_rates


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
