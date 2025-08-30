from typing import List
import torch
import plotly.graph_objects as go
import numpy as np


def visualize_losses(losses: List[torch.Tensor], title: str = "Training Losses by Epoch") -> go.Figure:
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

    # Generate distinct colors for each epoch
    colors = [
        f'hsl({(i * 360) / len(losses)}, 70%, 50%)'
        for i in range(len(losses))
    ]

    for epoch_idx, epoch_losses in enumerate(losses):
        # Convert tensor to numpy for plotting
        loss_values = epoch_losses.detach().cpu().numpy()
        batch_indices = np.arange(len(loss_values))

        fig.add_trace(go.Scatter(
            x=batch_indices,
            y=loss_values,
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
