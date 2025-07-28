"""
D3Feat Model Wrapper for Pylon API Compatibility.

This module provides a Pylon-compatible wrapper around the original D3Feat KPFCNN model.
"""

from typing import Dict, Any, List
import torch
import torch.nn as nn
from easydict import EasyDict

from models.point_cloud_registration.d3feat.architectures import _KPFCNN
from data.collators.d3feat.dataloader import collate_fn_descriptor, calibrate_neighbors


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
        self.neighborhood_limits = None
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with Pylon point cloud dictionary format.
        
        Args:
            inputs: Dictionary containing:
                - src_pc: Source point cloud dict with 'pos' and 'feat' keys
                - tgt_pc: Target point cloud dict with 'pos' and 'feat' keys  
                - correspondences: Ground truth correspondences (for training)
                
        Returns:
            Dictionary with:
                - descriptors: Combined descriptors [N_total, feature_dim]
                - scores: Combined detection scores [N_total, 1]
        """
        # Extract point clouds
        src_pc = inputs['src_pc']
        tgt_pc = inputs['tgt_pc']
        
        # Convert to D3Feat format (numpy arrays)
        pts0 = src_pc['pos'].detach().cpu().numpy()  # [N1, 3]
        pts1 = tgt_pc['pos'].detach().cpu().numpy()  # [N2, 3]
        feat0 = src_pc['feat'].detach().cpu().numpy()  # [N1, feat_dim]
        feat1 = tgt_pc['feat'].detach().cpu().numpy()  # [N2, feat_dim]
        
        # Create dummy correspondences and distances for collate function
        # In actual training, these would come from the dataset
        if 'correspondences' in inputs:
            corr = inputs['correspondences'].detach().cpu().numpy()
            # Compute distances between corresponding points for loss
            dist_keypts = torch.cdist(
                src_pc['pos'][corr[:, 0]], 
                src_pc['pos'][corr[:, 0]]
            ).detach().cpu().numpy()
        else:
            # Dummy values for inference
            corr = torch.empty((0, 2)).numpy()
            dist_keypts = torch.empty((0, 0)).numpy()
        
        # Create tuple format expected by collate function
        list_data = [(pts0, pts1, feat0, feat1, corr, dist_keypts)]
        
        # Calibrate neighborhood limits if needed
        if self.neighborhood_limits is None:
            # Use reasonable defaults for inference
            self.neighborhood_limits = [20] * self.config.num_layers
        
        # Apply collate function to get batch format
        batch = collate_fn_descriptor(list_data, self.config, self.neighborhood_limits)
        
        # Move batch to device
        device = src_pc['pos'].device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
            elif isinstance(batch[key], list):
                batch[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch[key]]
        
        # Run original D3Feat model
        features, scores = self.d3feat_model(batch)
        
        return {
            'descriptors': features,  # [N_total, feature_dim] 
            'scores': scores,         # [N_total, 1]
        }
        
    def set_neighborhood_limits(self, limits: List[int]) -> None:
        """Set neighborhood limits for KPConv operations.
        
        Args:
            limits: List of neighborhood limits per layer
        """
        self.neighborhood_limits = limits