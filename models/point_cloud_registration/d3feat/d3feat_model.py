"""
D3Feat Model Wrapper for Pylon API Compatibility.

This module provides a Pylon-compatible wrapper around the original D3Feat KPFCNN model.
"""

from typing import Dict, Any, List
import torch
import torch.nn as nn
from easydict import EasyDict

from models.point_cloud_registration.d3feat.architectures import _KPFCNN


class D3FeatModel(nn.Module):
    """Pylon API wrapper for D3Feat KPFCNN model.
    
    This wrapper adapts the original D3Feat model to work with Pylon's point cloud
    dictionary format and training infrastructure.
    """
    
    def __init__(
        self,
        num_layers: int = 5,
        in_points_dim: int = 3,
        first_features_dim: int = 128,
        first_subsampling_dl: float = 0.03,
        in_features_dim: int = 1,
        conv_radius: float = 2.5,
        deform_radius: float = 5.0,
        num_kernel_points: int = 15,
        KP_extent: float = 2.0,
        KP_influence: str = 'linear',
        aggregation_mode: str = 'sum',
        fixed_kernel_points: str = 'center',
        use_batch_norm: bool = False,
        batch_norm_momentum: float = 0.02,
        deformable: bool = False,
        modulated: bool = False,
        **kwargs
    ):
        """Initialize D3Feat model wrapper.
        
        Args:
            num_layers: Number of network layers
            in_points_dim: Input point dimension (3 for xyz)
            first_features_dim: First layer feature dimension
            first_subsampling_dl: First subsampling grid size
            in_features_dim: Input feature dimension
            conv_radius: Convolution radius
            deform_radius: Deformable convolution radius  
            num_kernel_points: Number of kernel points
            KP_extent: Kernel point extent
            KP_influence: Kernel point influence type
            aggregation_mode: Aggregation mode
            fixed_kernel_points: Fixed kernel points mode
            use_batch_norm: Whether to use batch normalization
            batch_norm_momentum: Batch norm momentum
            deformable: Whether to use deformable convolutions
            modulated: Whether to use modulated convolutions
        """
        super(D3FeatModel, self).__init__()
        
        # Build D3Feat configuration
        config = EasyDict()
        config.num_layers = num_layers
        config.in_points_dim = in_points_dim
        config.first_features_dim = first_features_dim
        config.first_subsampling_dl = first_subsampling_dl
        config.in_features_dim = in_features_dim
        config.conv_radius = conv_radius
        config.deform_radius = deform_radius
        config.num_kernel_points = num_kernel_points
        config.KP_extent = KP_extent
        config.KP_influence = KP_influence
        config.aggregation_mode = aggregation_mode
        config.fixed_kernel_points = fixed_kernel_points
        config.use_batch_norm = use_batch_norm
        config.batch_norm_momentum = batch_norm_momentum
        config.deformable = deformable
        config.modulated = modulated
        
        # Build network architecture configuration
        config.architecture = ['simple', 'resnetb']
        for i in range(config.num_layers-1):
            config.architecture.append('resnetb_strided')
            config.architecture.append('resnetb')
            config.architecture.append('resnetb')
        for i in range(config.num_layers-2):
            config.architecture.append('nearest_upsample')
            config.architecture.append('unary')
        config.architecture.append('nearest_upsample')
        config.architecture.append('last_unary')
        
        # Initialize the original D3Feat model
        self.d3feat_model = _KPFCNN(config)
        self.config = config
        
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass with collated batch inputs.
        
        Args:
            inputs: Dictionary containing collated batch data:
                - points: List of point tensors per layer
                - neighbors: List of neighbor indices per layer
                - pools: List of pooling indices per layer
                - upsamples: List of upsampling indices per layer
                - features: Batched features tensor
                - stack_lengths: List of stack lengths per layer
                - corr: Correspondence tensor
                - dist_keypts: Distance keypoints tensor
                
        Returns:
            Dictionary with:
                - descriptors: Combined descriptors [N_total, feature_dim]
                - scores: Combined detection scores [N_total, 1]
        """
        # Run original D3Feat model with collated inputs
        features, scores = self.d3feat_model(inputs)
        
        return {
            'descriptors': features,        # [N_total, feature_dim] 
            'scores': scores,               # [N_total, 1]
            'stack_lengths': inputs['stack_lengths'],  # Pass through for criterion
        }
