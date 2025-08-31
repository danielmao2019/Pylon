from typing import List
import torch
import plotly.graph_objects as go
import numpy as np

from runners.viewers.train_viewer.backend.utils import apply_smoothing


def visualize_losses(losses: List[torch.Tensor], smoothing_window: int = 1, title: str = "Training Losses by Epoch") -> go.Figure:
    """Create a plotly figure visualizing training losses across epochs.

    Args:
        losses: List of loss tensors, one per epoch
        title: Title for the plot

    Returns:
        Plotly figure with loss curves, different color for each epoch
    """
    # Input validation following CLAUDE.md fail-fast patterns
    assert losses is not None, "losses must not be None"
    assert isinstance(losses, list), f"losses must be list, got {type(losses)}"
    assert len(losses) > 0, f"losses must not be empty"
    assert all(isinstance(loss, torch.Tensor) for loss in losses), "All losses must be torch.Tensor"

    fig = go.Figure()

    # Generate smooth color transition across epochs
    # Start from red (0°) and smoothly transition through the color wheel
    colors = [
        f'hsl({(i * 300) / max(len(losses) - 1, 1)}, 70%, 50%)'
        for i in range(len(losses))
    ]

    # Concatenate all losses and apply smoothing on the full sequence
    all_losses = torch.cat(losses)
    all_loss_values = all_losses.detach().cpu().numpy()

    # Apply smoothing on concatenated losses
    smoothed_losses = apply_smoothing(all_loss_values, smoothing_window)

    # Track start and end indices for each epoch
    epoch_boundaries = []
    current_idx = 0
    for epoch_losses in losses:
        epoch_length = len(epoch_losses)
        epoch_boundaries.append((current_idx, current_idx + epoch_length))
        current_idx += epoch_length

    # Plot each epoch with its proper indices
    for epoch_idx, (start_idx, end_idx) in enumerate(epoch_boundaries):
        # Extract smoothed values for this epoch
        epoch_smoothed_values = smoothed_losses[start_idx:end_idx]

        # Use start_idx to end_idx directly as batch indices
        batch_indices = np.arange(start_idx, end_idx)

        fig.add_trace(go.Scatter(
            x=batch_indices,
            y=epoch_smoothed_values,
            mode='lines+markers',
            name=f'Epoch {epoch_idx}',
            line=dict(color=colors[epoch_idx]),
            marker=dict(size=4)
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Batch Index',
        yaxis_title='Loss Value',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )

    return fig
