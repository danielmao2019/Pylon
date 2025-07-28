"""
D3Feat Collator for Pylon Integration.

This module provides a collator that can work with Pylon's existing ThreeDMatch dataset
to prepare data in the format expected by D3Feat models.
"""

from typing import List, Dict, Any, Tuple
import torch
import numpy as np
from functools import partial

from data.collators.base_collator import BaseCollator
from data.collators.d3feat.dataloader import (
    collate_fn_descriptor, calibrate_neighbors
)


class D3FeatCollator(BaseCollator):
    """Collator for D3Feat that bridges Pylon format and D3Feat internal format.
    
    This collator takes Pylon's three-field format and converts it to the hierarchical
    batch format expected by D3Feat's KPConv operations.
    """
    
    def __init__(
        self,
        num_layers: int = 5,
        first_subsampling_dl: float = 0.03,
        conv_radius: float = 2.5,
        deform_radius: float = 5.0,
        num_kernel_points: int = 15,
        neighborhood_limits: List[int] = None,
        **kwargs
    ):
        """Initialize D3Feat collator.
        
        Args:
            num_layers: Number of network layers
            first_subsampling_dl: First subsampling grid size
            conv_radius: Convolution radius
            deform_radius: Deformable convolution radius
            num_kernel_points: Number of kernel points
            neighborhood_limits: Pre-computed neighborhood limits per layer
        """
        super(D3FeatCollator, self).__init__(**kwargs)
        
        self.num_layers = num_layers
        self.first_subsampling_dl = first_subsampling_dl
        self.conv_radius = conv_radius
        self.deform_radius = deform_radius
        self.num_kernel_points = num_kernel_points
        self.neighborhood_limits = neighborhood_limits
        
        # Build architecture configuration for collate function
        self.architecture = ['simple', 'resnetb']
        for i in range(num_layers-1):
            self.architecture.append('resnetb_strided')
            self.architecture.append('resnetb')
            self.architecture.append('resnetb')
        for i in range(num_layers-2):
            self.architecture.append('nearest_upsample')
            self.architecture.append('unary')
        self.architecture.append('nearest_upsample')
        self.architecture.append('last_unary')
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch of Pylon format data into D3Feat format.
        
        Args:
            batch: List of datapoints in Pylon three-field format
            
        Returns:
            Collated batch in D3Feat hierarchical format
        """
        # Convert Pylon format to D3Feat tuple format
        list_data = []
        
        for datapoint in batch:
            inputs = datapoint['inputs']
            labels = datapoint['labels'] 
            meta_info = datapoint['meta_info']
            
            # Extract point clouds
            src_pc = inputs['src_pc']
            tgt_pc = inputs['tgt_pc']
            
            # Convert to numpy arrays (D3Feat expects numpy)
            pts0 = src_pc['pos'].detach().cpu().numpy().astype(np.float32)
            pts1 = tgt_pc['pos'].detach().cpu().numpy().astype(np.float32)
            feat0 = src_pc['feat'].detach().cpu().numpy().astype(np.float32)
            feat1 = tgt_pc['feat'].detach().cpu().numpy().astype(np.float32)
            
            # Get correspondences
            if 'correspondences' in inputs:
                correspondences = inputs['correspondences'].detach().cpu().numpy().astype(np.int32)
            else:
                correspondences = np.array([], dtype=np.int32).reshape(0, 2)
                
            # Compute distances between keypoints (simplified)
            if correspondences.shape[0] > 0:
                # Get corresponding points
                corr_pts_src = pts0[correspondences[:, 0]]
                corr_pts_tgt = pts1[correspondences[:, 1]]
                
                # Compute pairwise distances between corresponding source points
                from scipy.spatial.distance import cdist
                dist_keypts = cdist(corr_pts_src, corr_pts_src).astype(np.float32)
            else:
                dist_keypts = np.array([], dtype=np.float32).reshape(0, 0)
            
            # Create tuple format expected by D3Feat collate function
            list_data.append((pts0, pts1, feat0, feat1, correspondences, dist_keypts))
        
        # Use default neighborhood limits if not provided
        if self.neighborhood_limits is None:
            self.neighborhood_limits = [20] * self.num_layers
            
        # Create config object for collate function
        config = type('Config', (), {})()
        config.first_subsampling_dl = self.first_subsampling_dl
        config.conv_radius = self.conv_radius
        config.deform_radius = self.deform_radius
        config.architecture = self.architecture
        
        # Apply D3Feat collate function
        d3feat_batch = collate_fn_descriptor(list_data, config, self.neighborhood_limits)
        
        # Add original Pylon data for potential use in loss computation
        d3feat_batch['pylon_batch'] = batch
        
        return d3feat_batch
        
    def calibrate_neighborhood_limits(self, dataset, keep_ratio: float = 0.8, samples_threshold: int = 2000):
        """Calibrate neighborhood limits for efficient KPConv operations.
        
        Args:
            dataset: Dataset to calibrate on
            keep_ratio: Ratio of neighbors to keep
            samples_threshold: Number of samples for calibration
            
        Returns:
            Calibrated neighborhood limits
        """
        # Create config for calibration
        config = type('Config', (), {})()
        config.first_subsampling_dl = self.first_subsampling_dl
        config.conv_radius = self.conv_radius
        config.deform_radius = self.deform_radius
        config.num_layers = self.num_layers
        config.architecture = self.architecture
        
        # Create temporary collator for calibration
        temp_collator = partial(collate_fn_descriptor, config=config)
        
        # Calibrate using D3Feat's calibration function
        self.neighborhood_limits = calibrate_neighbors(
            dataset, config, temp_collator, keep_ratio, samples_threshold
        )
        
        return self.neighborhood_limits