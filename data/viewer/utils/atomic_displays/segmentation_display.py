"""Segmentation display utilities for semantic/instance segmentation visualization."""
from typing import Dict, Union, Any, Optional, List
import torch
import plotly.graph_objects as go
from data.viewer.utils.segmentation import create_segmentation_figure, get_segmentation_stats


def create_segmentation_display(
    segmentation: Union[torch.Tensor, Dict[str, Any]],
    title: str,
    class_labels: Optional[Dict[str, List[str]]] = None,
    color_seed: int = 0,
    **kwargs: Any
) -> go.Figure:
    """Create segmentation display for semantic or instance segmentation.
    
    Args:
        segmentation: Segmentation data, can be:
            - 2D tensor of shape [H, W] or [N, H, W] (batched) with class indices
            - Dict with keys "masks" (List[torch.Tensor]) and "indices" (List[Any])
        title: Title for the segmentation display
        class_labels: Optional mapping from class indices to label names
        color_seed: Seed for color generation to shuffle colors (default: 0)
        **kwargs: Additional arguments passed to create_segmentation_figure
        
    Returns:
        Plotly figure for segmentation visualization
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # CRITICAL: Input validation with fail-fast assertions
    assert isinstance(title, str), f"Expected str title, got {type(title)}"
    
    if isinstance(segmentation, torch.Tensor):
        assert segmentation.ndim in [2, 3], f"Expected 2D [H,W] or 3D [N,H,W] tensor, got shape {segmentation.shape}"
        assert segmentation.numel() > 0, f"Segmentation tensor cannot be empty"
        assert segmentation.dtype in [torch.int64, torch.long], f"Expected int64 segmentation, got {segmentation.dtype}"
        
        # Handle batched input - extract single sample for visualization  
        if segmentation.ndim == 3:
            assert segmentation.shape[0] == 1, f"Expected batch size 1 for visualization, got {segmentation.shape[0]}"
            segmentation = segmentation[0]  # [N, H, W] -> [H, W]
            
    elif isinstance(segmentation, dict):
        assert 'masks' in segmentation, f"Dict segmentation must have 'masks', got keys: {list(segmentation.keys())}"
        assert 'indices' in segmentation, f"Dict segmentation must have 'indices', got keys: {list(segmentation.keys())}"
        assert isinstance(segmentation['masks'], list), f"masks must be list, got {type(segmentation['masks'])}"
        assert isinstance(segmentation['indices'], list), f"indices must be list, got {type(segmentation['indices'])}"
        assert len(segmentation['masks']) > 0, f"masks cannot be empty"
        assert len(segmentation['masks']) == len(segmentation['indices']), \
            f"masks and indices must have same length, got {len(segmentation['masks'])} vs {len(segmentation['indices'])}"
    else:
        assert False, f"segmentation must be torch.Tensor or dict, got {type(segmentation)}"
    
    if class_labels is not None:
        assert isinstance(class_labels, dict), f"class_labels must be dict, got {type(class_labels)}"
    
    # Use existing create_segmentation_figure implementation with color seed
    return create_segmentation_figure(seg=segmentation, title=title, color_seed=color_seed)


def get_segmentation_display_stats(
    segmentation: Union[torch.Tensor, Dict[str, Any]],
    color_seed: int = 0
) -> Dict[str, Any]:
    """Get segmentation statistics for display.
    
    Args:
        segmentation: Segmentation data, can be:
            - 2D tensor of shape [H, W] or [N, H, W] (batched) with class indices
            - Dict with keys "masks" (List[torch.Tensor]) and "indices" (List[Any])
        color_seed: Seed for color generation to shuffle colors (default: 0)
        
    Returns:
        Dictionary containing segmentation statistics
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validation and batch handling
    if isinstance(segmentation, torch.Tensor):
        assert segmentation.ndim in [2, 3], f"Expected 2D [H,W] or 3D [N,H,W] tensor, got shape {segmentation.shape}"
        assert segmentation.numel() > 0, f"Segmentation tensor cannot be empty"
        
        # Handle batched input - extract single sample for analysis
        if segmentation.ndim == 3:
            assert segmentation.shape[0] == 1, f"Expected batch size 1 for analysis, got {segmentation.shape[0]}"
            segmentation = segmentation[0]  # [N, H, W] -> [H, W]
            
    elif isinstance(segmentation, dict):
        assert 'masks' in segmentation, f"Dict segmentation must have 'masks', got keys: {list(segmentation.keys())}"
        assert 'indices' in segmentation, f"Dict segmentation must have 'indices', got keys: {list(segmentation.keys())}"
    else:
        assert False, f"segmentation must be torch.Tensor or dict, got {type(segmentation)}"
    
    # Use existing get_segmentation_stats implementation with color seed
    return get_segmentation_stats(segmentation, color_seed=color_seed)
