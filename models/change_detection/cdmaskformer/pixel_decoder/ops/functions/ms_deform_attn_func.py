"""
MultiScale Deformable Attention Function.

This implementation is a PyTorch-only version without custom CUDA implementation.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Multi-scale deformable attention (PyTorch implementation).
    
    Args:
        value (torch.Tensor): The value tensor with shape (N, Length, C).
        value_spatial_shapes (torch.Tensor): Spatial shapes of the value features.
        sampling_locations (torch.Tensor): Sampling locations with shape (N, Length, n_heads, n_levels, n_points, 2).
        attention_weights (torch.Tensor): Attention weights with shape (N, Length, n_heads, n_levels, n_points).
        
    Returns:
        torch.Tensor: Attention output with shape (N, Length, C).
    """
    N_, S_, M_, h_, P_, _ = sampling_locations.shape
    # (N_, S_, M_, h_, P_) -> (N_, h_, S_, M_, P_) -> (N_*h_, S_, M_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * h_, S_, M_ * P_)
    sampling_locations = sampling_locations.transpose(1, 2).reshape(N_ * h_, S_, M_ * P_, 2)
    
    # Process for each batch and head
    output = torch.zeros_like(value)
    output = output.reshape(N_ * h_, S_, -1)
    
    # Get value features
    value_list = []
    level_start_index = torch.cat((value_spatial_shapes.new_zeros((1,)), value_spatial_shapes.prod(1).cumsum(0)[:-1]))
    
    for lv_idx in range(len(value_spatial_shapes)):
        h_lv, w_lv = value_spatial_shapes[lv_idx]
        level_value = value[:, level_start_index[lv_idx]:level_start_index[lv_idx] + h_lv * w_lv, :]
        level_value = level_value.reshape(N_, h_lv, w_lv, -1).permute(0, 3, 1, 2)  # (N_, C, h_lv, w_lv)
        value_list.append(level_value)
    
    # Apply attention with bilinear sampling
    for lv_idx, level_value in enumerate(value_list):
        h_lv, w_lv = value_spatial_shapes[lv_idx]
        
        # Get sampling locations for this level
        level_sampling_loc = sampling_locations[:, :, lv_idx*P_:(lv_idx+1)*P_, :]  # (N_*h_, S_, P_, 2)
        level_sampling_loc = level_sampling_loc.reshape(N_, h_, S_, P_, 2).transpose(1, 2).reshape(N_*S_, h_, P_, 2)
        
        # Get attention weights for this level
        level_attn_weight = attention_weights[:, :, lv_idx*P_:(lv_idx+1)*P_]  # (N_*h_, S_, P_)
        level_attn_weight = level_attn_weight.reshape(N_, h_, S_, P_).transpose(1, 2).reshape(N_*S_, h_, P_)
        
        # Apply sampling
        level_value = level_value.reshape(N_, 1, -1, h_lv, w_lv).expand(N_, h_, -1, h_lv, w_lv)
        level_value = level_value.reshape(N_*h_, -1, h_lv, w_lv)
        
        # Perform bilinear sampling
        # Normalize coordinates to [-1, 1]
        normalized_loc = level_sampling_loc.clone()
        normalized_loc[..., 0] = 2 * (normalized_loc[..., 0] / (w_lv - 1)) - 1
        normalized_loc[..., 1] = 2 * (normalized_loc[..., 1] / (h_lv - 1)) - 1
        
        # Sample with grid_sample
        sampled_feats = F.grid_sample(
            level_value, 
            normalized_loc, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )  # (N_*h_, C/h_, S_, P_)
        
        # Apply attention weights
        sampled_feats = sampled_feats.reshape(N_, h_, -1, S_, P_)
        sampled_feats = sampled_feats.permute(0, 3, 1, 4, 2).reshape(N_*S_, h_, P_, -1)
        weighted_feats = (sampled_feats * level_attn_weight.unsqueeze(-1)).sum(dim=2)  # (N_*S_, h_, C/h_)
        weighted_feats = weighted_feats.reshape(N_, S_, h_, -1).permute(0, 2, 1, 3).reshape(N_*h_, S_, -1)
        
        # Add to output
        output = output + weighted_feats
    
    output = output.reshape(N_, h_, S_, -1).permute(0, 2, 1, 3).reshape(N_, S_, -1)
    return output
