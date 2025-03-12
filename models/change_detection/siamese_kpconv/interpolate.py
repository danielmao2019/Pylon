"""
Interpolation operations for SiameseKPConv model.

This module provides interpolation-related functionality, including KNN interpolation
and scatter operations used in feature propagation.
"""
import torch
from models.change_detection.siamese_kpconv.utils import knn


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> torch.Tensor:
    """Pure PyTorch implementation of scatter_add operation.
    
    Args:
        src: Source tensor to scatter
        index: Index tensor indicating where to scatter
        dim: Dimension along which to scatter
        dim_size: Size of the output tensor along scatter dimension
        
    Returns:
        Output tensor with scattered and added values
    """
    # Create output tensor filled with zeros
    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
    
    # Use index_add_ which is PyTorch's built-in scatter add operation
    return out.index_add_(0, index, src)


class KNNInterpolate:
    """KNN interpolation module for feature propagation in the decoder.
    Matches the original implementation's upsampling approach."""
    
    def __init__(self, k: int = 3):
        self.k = k
    
    def __call__(self, query_pos: torch.Tensor, query_batch: torch.Tensor,
                 support_pos: torch.Tensor, support_batch: torch.Tensor,
                 support_features: torch.Tensor) -> torch.Tensor:
        """
        Interpolate features from support points to query points using KNN.
        
        Args:
            query_pos: [N, 3] query point positions
            query_batch: [N] query point batch indices
            support_pos: [M, 3] support point positions
            support_batch: [M] support point batch indices
            support_features: [M, C] support point features
            
        Returns:
            Interpolated features [N, C]
        """
        # Find k nearest neighbors
        row_idx, col_idx = knn(query_pos, support_pos, self.k, query_batch, support_batch)
        
        # Get squared distances
        diff = query_pos[row_idx] - support_pos[col_idx]
        squared_dist = (diff * diff).sum(dim=1, keepdim=True)
        
        # Compute weights with numerical stability
        weights = 1.0 / torch.clamp(squared_dist, min=1e-16)
        
        # Normalize weights
        normalizer = scatter_add(weights, row_idx, 0, query_pos.size(0))
        weights = weights / normalizer[row_idx]
        
        # Interpolate features
        features = support_features[col_idx]
        weighted_features = features * weights
        interpolated = scatter_add(weighted_features, row_idx, 0, query_pos.size(0))
        
        return interpolated 