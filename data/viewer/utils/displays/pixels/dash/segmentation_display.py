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
    assert isinstance(segmentation, (torch.Tensor, dict)), (
        "segmentation must be torch.Tensor or dict. " f"{type(segmentation)=}"
    )
    assert isinstance(title, str), f"Expected str title. {type(title)=}"
    assert class_labels is None or isinstance(
        class_labels, dict
    ), f"class_labels must be dict when provided. {type(class_labels)=}"
    if isinstance(segmentation, torch.Tensor):
        assert segmentation.ndim in [2, 3], (
            "Expected 2D [H,W] or 3D [N,H,W] tensor. " f"{segmentation.shape=}"
        )
        assert (
            segmentation.numel() > 0
        ), f"Segmentation tensor cannot be empty. {segmentation.shape=}"
        assert segmentation.dtype in [torch.int64, torch.long], (
            "Expected int64 segmentation tensor. " f"{segmentation.dtype=}"
        )
        assert segmentation.ndim != 3 or segmentation.shape[0] == 1, (
            "Expected batch size 1 for visualization. " f"{segmentation.shape=}"
        )
    if isinstance(segmentation, dict):
        assert "masks" in segmentation, (
            "Dict segmentation must have 'masks'. " f"{segmentation.keys()=}"
        )
        assert "indices" in segmentation, (
            "Dict segmentation must have 'indices'. " f"{segmentation.keys()=}"
        )
        assert isinstance(segmentation["masks"], list), (
            "Dict segmentation masks must be list. " f"{type(segmentation['masks'])=}"
        )
        assert isinstance(segmentation["indices"], list), (
            "Dict segmentation indices must be list. "
            f"{type(segmentation['indices'])=}"
        )
        assert len(segmentation["masks"]) > 0, (
            "Dict segmentation masks cannot be empty. " f"{len(segmentation['masks'])=}"
        )
        assert len(segmentation["masks"]) == len(segmentation["indices"]), (
            "Dict segmentation masks and indices must have same length. "
            f"{len(segmentation['masks'])=} {len(segmentation['indices'])=}"
        )

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
    assert isinstance(segmentation, (torch.Tensor, dict)), (
        "segmentation must be torch.Tensor or dict. " f"{type(segmentation)=}"
    )
    if isinstance(segmentation, torch.Tensor):
        assert segmentation.ndim in [2, 3], (
            "Expected 2D [H,W] or 3D [N,H,W] tensor. " f"{segmentation.shape=}"
        )
        assert (
            segmentation.numel() > 0
        ), f"Segmentation tensor cannot be empty. {segmentation.shape=}"
        assert segmentation.ndim != 3 or segmentation.shape[0] == 1, (
            "Expected batch size 1 for analysis. " f"{segmentation.shape=}"
        )
    if isinstance(segmentation, dict):
        assert "masks" in segmentation, (
            "Dict segmentation must have 'masks'. " f"{segmentation.keys()=}"
        )
        assert "indices" in segmentation, (
            "Dict segmentation must have 'indices'. " f"{segmentation.keys()=}"
        )

    # Input normalizations
    if isinstance(segmentation, torch.Tensor) and segmentation.ndim == 3:
        segmentation = segmentation[0]  # [N, H, W] -> [H, W]

    # Use existing get_segmentation_stats implementation
    return get_segmentation_stats(segmentation)
