import numpy as np


def apply_smoothing(losses: np.ndarray, window_size: int) -> np.ndarray:
    """Apply moving average smoothing to loss values.

    Args:
        losses: 1D array of loss values
        window_size: Size of the moving average window

    Returns:
        Smoothed loss values (same length as input)
    """
    assert losses is not None, "losses must not be None"
    assert isinstance(losses, np.ndarray), f"losses must be numpy array, got {type(losses)}"
    assert losses.ndim == 1, f"losses must be 1D array, got {losses.ndim}D"
    assert window_size >= 1, f"window_size must be >= 1, got {window_size}"

    if window_size == 1:
        return losses

    # Use numpy's convolve for moving average
    kernel = np.ones(window_size) / window_size
    # Use 'same' mode to keep same length, pad at edges
    smoothed = np.convolve(losses, kernel, mode='same')

    return smoothed
