"""Segmentation display utilities for semantic/instance segmentation visualization."""

from typing import Any, Dict, List, Optional, Union

import plotly.graph_objects as go
import torch

from data.viewer.utils.segmentation import (
    create_segmentation_figure,
    get_segmentation_stats,
)


def create_segmentation_display(
    segmentation: Union[torch.Tensor, Dict[str, Any]],
    title: str,
    class_labels: Optional[Dict[str, List[str]]] = None,
    **kwargs: Any,
) -> go.Figure:
    """Create segmentation display for semantic or instance segmentation.

    Args:
        segmentation: Segmentation data, can be:
            - 2D tensor of shape [H, W] or [N, H, W] (batched) with class indices
            - Dict with keys "masks" (List[torch.Tensor]) and "indices" (List[Any])
        title: Title for the segmentation display
        class_labels: Optional mapping from class indices to label names
        **kwargs: Additional arguments passed to create_segmentation_figure

    Returns:
        Plotly figure for segmentation visualization

    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validations
    assert isinstance(segmentation, (torch.Tensor, dict)), f"{type(segmentation)=}"
    assert isinstance(title, str), f"{type(title)=}"
    assert class_labels is None or isinstance(
        class_labels, dict
    ), f"{type(class_labels)=}"
    assert not isinstance(segmentation, torch.Tensor) or (
        segmentation.ndim in [2, 3]
        and segmentation.numel() > 0
        and segmentation.dtype in [torch.int64, torch.long]
        and (segmentation.ndim != 3 or segmentation.shape[0] == 1)
    ), f"{segmentation.shape=}"
    assert not isinstance(segmentation, dict) or (
        "masks" in segmentation
        and "indices" in segmentation
        and isinstance(segmentation["masks"], list)
        and isinstance(segmentation["indices"], list)
        and len(segmentation["masks"]) > 0
        and len(segmentation["masks"]) == len(segmentation["indices"])
    ), f"{segmentation=}"

    # Input normalizations
    if isinstance(segmentation, torch.Tensor) and segmentation.ndim == 3:
        segmentation = segmentation[0]  # [N, H, W] -> [H, W]

    # Use existing create_segmentation_figure implementation
    return create_segmentation_figure(seg=segmentation, title=title)


def get_segmentation_display_stats(
    segmentation: Union[torch.Tensor, Dict[str, Any]],
) -> Dict[str, Any]:
    """Get segmentation statistics for display.

    Args:
        segmentation: Segmentation data, can be:
            - 2D tensor of shape [H, W] or [N, H, W] (batched) with class indices
            - Dict with keys "masks" (List[torch.Tensor]) and "indices" (List[Any])

    Returns:
        Dictionary containing segmentation statistics

    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validations
    assert isinstance(segmentation, (torch.Tensor, dict)), f"{type(segmentation)=}"
    assert not isinstance(segmentation, torch.Tensor) or (
        segmentation.ndim in [2, 3]
        and segmentation.numel() > 0
        and (segmentation.ndim != 3 or segmentation.shape[0] == 1)
    ), f"{segmentation.shape=}"
    assert not isinstance(segmentation, dict) or (
        "masks" in segmentation and "indices" in segmentation
    ), f"{segmentation.keys()=}"

    # Input normalizations
    if isinstance(segmentation, torch.Tensor) and segmentation.ndim == 3:
        segmentation = segmentation[0]  # [N, H, W] -> [H, W]

    # Use existing get_segmentation_stats implementation
    return get_segmentation_stats(segmentation)
