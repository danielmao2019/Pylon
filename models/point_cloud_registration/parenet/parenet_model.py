"""
PARENet Model Wrapper for Pylon API Compatibility.

This module provides a Pylon-compatible wrapper around the original PARENet model.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from easydict import EasyDict

from models.point_cloud_registration.parenet.model import _PARE_Net


class PARENetModel(nn.Module):
    """Pylon API wrapper for PARENet model.
    
    This wrapper adapts the original PARENet model to work with Pylon's point cloud
    dictionary format and training infrastructure.
    """
    
    def __init__(
        self,
        # Model architecture parameters
        num_points_in_patch: int = 64,
        ground_truth_matching_radius: float = 0.05,
        
        # Backbone parameters
        backbone_init_dim: int = 3,
        backbone_output_dim: int = 96,
        backbone_kernel_size: float = 15,
        backbone_share_nonlinearity: bool = False,
        backbone_conv_way: str = 'standard',
        backbone_use_xyz: bool = False,
        
        # Fine matching parameters
        use_encoder_re_feats: bool = True,
        
        # GeoTransformer parameters
        geotransformer_input_dim: int = 96,
        geotransformer_output_dim: int = 96,
        geotransformer_hidden_dim: int = 128,
        geotransformer_num_heads: int = 4,
        geotransformer_blocks: list = None,
        geotransformer_sigma_d: float = 0.2,
        geotransformer_sigma_a: float = 15,
        geotransformer_angle_k: int = 3,
        geotransformer_reduction_a: str = 'max',
        
        # Coarse matching parameters
        coarse_matching_num_targets: int = 128,
        coarse_matching_overlap_threshold: float = 0.1,
        coarse_matching_num_correspondences: int = 256,
        coarse_matching_dual_normalization: bool = True,
        
        # Fine matching parameters
        fine_matching_topk: int = 3,
        fine_matching_acceptance_radius: float = 0.1,
        fine_matching_confidence_threshold: float = 0.05,
        fine_matching_num_hypotheses: int = 1000,
        fine_matching_num_refinement_steps: int = 5,
        
        **kwargs
    ):
        """Initialize PARENet model wrapper.
        
        Args:
            num_points_in_patch: Number of points in each patch
            ground_truth_matching_radius: Radius for ground truth matching
            backbone_init_dim: Backbone initial dimension
            backbone_output_dim: Backbone output dimension
            backbone_kernel_size: Backbone kernel size
            backbone_share_nonlinearity: Whether to share nonlinearity in backbone
            backbone_conv_way: Convolution method ('standard' or other)
            backbone_use_xyz: Whether to use xyz coordinates
            use_encoder_re_feats: Whether to use encoder RE features
            geotransformer_input_dim: GeoTransformer input dimension
            geotransformer_output_dim: GeoTransformer output dimension
            geotransformer_hidden_dim: GeoTransformer hidden dimension
            geotransformer_num_heads: Number of attention heads
            geotransformer_blocks: List of transformer blocks
            geotransformer_sigma_d: Sigma for distance
            geotransformer_sigma_a: Sigma for angle
            geotransformer_angle_k: Angle k parameter
            geotransformer_reduction_a: Reduction method for angle
            coarse_matching_num_targets: Number of coarse matching targets
            coarse_matching_overlap_threshold: Overlap threshold for coarse matching
            coarse_matching_num_correspondences: Number of correspondences
            coarse_matching_dual_normalization: Whether to use dual normalization
            fine_matching_topk: Top-k for fine matching
            fine_matching_acceptance_radius: Acceptance radius for fine matching
            fine_matching_confidence_threshold: Confidence threshold
            fine_matching_num_hypotheses: Number of hypotheses
            fine_matching_num_refinement_steps: Number of refinement steps
        """
        super(PARENetModel, self).__init__()
        
        # Set default blocks if not provided
        if geotransformer_blocks is None:
            geotransformer_blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
        
        # Build PARENet configuration using EasyDict for nested access
        cfg = EasyDict()
        
        # Model configuration
        cfg.model = EasyDict()
        cfg.model.num_points_in_patch = num_points_in_patch
        cfg.model.ground_truth_matching_radius = ground_truth_matching_radius
        
        # Backbone configuration
        cfg.backbone = EasyDict()
        cfg.backbone.init_dim = backbone_init_dim
        cfg.backbone.output_dim = backbone_output_dim
        cfg.backbone.kernel_size = backbone_kernel_size
        cfg.backbone.share_nonlinearity = backbone_share_nonlinearity
        cfg.backbone.conv_way = backbone_conv_way
        cfg.backbone.use_xyz = backbone_use_xyz
        
        # Fine matching configuration
        cfg.fine_matching = EasyDict()
        cfg.fine_matching.use_encoder_re_feats = use_encoder_re_feats
        cfg.fine_matching.topk = fine_matching_topk
        cfg.fine_matching.acceptance_radius = fine_matching_acceptance_radius
        cfg.fine_matching.confidence_threshold = fine_matching_confidence_threshold
        cfg.fine_matching.num_hypotheses = fine_matching_num_hypotheses
        cfg.fine_matching.num_refinement_steps = fine_matching_num_refinement_steps
        
        # GeoTransformer configuration
        cfg.geotransformer = EasyDict()
        cfg.geotransformer.input_dim = geotransformer_input_dim
        cfg.geotransformer.output_dim = geotransformer_output_dim
        cfg.geotransformer.hidden_dim = geotransformer_hidden_dim
        cfg.geotransformer.num_heads = geotransformer_num_heads
        cfg.geotransformer.blocks = geotransformer_blocks
        cfg.geotransformer.sigma_d = geotransformer_sigma_d
        cfg.geotransformer.sigma_a = geotransformer_sigma_a
        cfg.geotransformer.angle_k = geotransformer_angle_k
        cfg.geotransformer.reduction_a = geotransformer_reduction_a
        
        # Coarse matching configuration
        cfg.coarse_matching = EasyDict()
        cfg.coarse_matching.num_targets = coarse_matching_num_targets
        cfg.coarse_matching.overlap_threshold = coarse_matching_overlap_threshold
        cfg.coarse_matching.num_correspondences = coarse_matching_num_correspondences
        cfg.coarse_matching.dual_normalization = coarse_matching_dual_normalization
        
        # Initialize the original PARENet model
        self.parenet_model = _PARE_Net(cfg)
        self.cfg = cfg
        
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass with Pylon collated batch inputs.
        
        Args:
            inputs: Dictionary containing collated batch data from PARENetCollator:
                - points: List of point tensors per layer
                - lengths: List of length tensors per layer  
                - features: Batched features tensor
                - transform: Ground truth transformation matrix
                - batch_size: Batch size
                
        Returns:
            Dictionary with all outputs from the original PARENet model
        """
        # Run original PARENet model and return all outputs
        # No filtering needed - the original model produces the correct outputs
        return self.parenet_model(inputs)
