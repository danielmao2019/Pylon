"""
PARENet Collator for Pylon API Compatibility.

This module provides Pylon-compatible wrapper around PARENet's stack-mode collation.
"""

from typing import List, Dict, Any, Optional
import torch

from data.collators.base_collator import BaseCollator
from data.collators.parenet.data import registration_collate_fn_stack_mode


class PARENetCollator(BaseCollator):
    """Pylon wrapper for PARENet stack-mode collation."""
    
    def __init__(
        self,
        # Stack-mode parameters
        num_stages: int = 4,
        voxel_size: float = 0.05,
        subsample_ratio: float = 4.0,
        num_neighbors: Optional[List[int]] = None,
        
        # Base collator parameters
        collators: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Initialize PARENet collator.
        
        Args:
            num_stages: Number of hierarchical stages
            voxel_size: Base voxel size for subsampling
            subsample_ratio: Ratio between consecutive stages
            num_neighbors: List of neighbor counts for each stage
            collators: Custom collation functions for specific keys
        """
        super(PARENetCollator, self).__init__(collators=collators)
        
        # Store PARENet-specific parameters
        self.num_stages = num_stages
        self.voxel_size = voxel_size  
        self.subsample_ratio = subsample_ratio
        
        # Initialize default neighbor counts if not provided
        if num_neighbors is None:
            # Default values from PARENet
            self.num_neighbors = [32, 32, 32, 32][:num_stages]
        else:
            assert len(num_neighbors) == num_stages, f"num_neighbors length ({len(num_neighbors)}) must match num_stages ({num_stages})"
            self.num_neighbors = num_neighbors
        
    def __call__(self, datapoints: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Collate datapoints using PARENet stack-mode approach.
        
        Args:
            datapoints: List of Pylon datapoints with 'inputs', 'labels', 'meta_info'
            
        Returns:
            Collated dictionary with PARENet format plus Pylon structure
        """
        # Convert Pylon datapoints to PARENet format
        parenet_datapoints = []
        
        for datapoint in datapoints:
            # Extract inputs and labels
            inputs = datapoint['inputs']
            labels = datapoint['labels'] 
            meta_info = datapoint.get('meta_info', {})
            
            # Convert to PARENet expected format
            parenet_dict = {}
            
            # Handle point clouds - PARENet expects 'ref_points' and 'src_points'
            if 'src_pc' in inputs and 'tgt_pc' in inputs:
                src_pc = inputs['src_pc']
                tgt_pc = inputs['tgt_pc']
                
                # Handle different input formats
                if isinstance(src_pc, dict) and 'pos' in src_pc:
                    src_points = src_pc['pos']
                else:
                    src_points = src_pc
                    
                if isinstance(tgt_pc, dict) and 'pos' in tgt_pc:
                    tgt_points = tgt_pc['pos']
                else:
                    tgt_points = tgt_pc
                
                parenet_dict['ref_points'] = tgt_points  # PARENet uses ref for target
                parenet_dict['src_points'] = src_points
                
                # Handle features - use dummy features if not provided
                if 'src_features' in inputs and 'tgt_features' in inputs:
                    parenet_dict['src_feats'] = inputs['src_features']
                    parenet_dict['ref_feats'] = inputs['tgt_features']
                else:
                    # Generate dummy unit features for PARENet
                    parenet_dict['src_feats'] = torch.ones(src_points.shape[0], 1)
                    parenet_dict['ref_feats'] = torch.ones(tgt_points.shape[0], 1)
            
            # Handle transformation matrix
            if 'transform' in labels:
                parenet_dict['transform'] = labels['transform']
            
            # Include meta info
            parenet_dict.update(meta_info)
            
            parenet_datapoints.append(parenet_dict)
        
        # Use PARENet's original collation function
        parenet_collated = registration_collate_fn_stack_mode(
            parenet_datapoints,
            num_stages=self.num_stages,
            voxel_size=self.voxel_size,
            num_neighbors=self.num_neighbors,
            subsample_ratio=self.subsample_ratio
        )
        
        # Convert back to Pylon format
        pylon_collated = {
            'inputs': {},
            'labels': {},
            'meta_info': {}
        }
        
        # Map PARENet outputs to Pylon inputs
        if 'ref_points' in parenet_collated:
            pylon_collated['inputs']['tgt_pc'] = parenet_collated['ref_points']
        if 'src_points' in parenet_collated:
            pylon_collated['inputs']['src_pc'] = parenet_collated['src_points']
            
        # Copy all other PARENet-specific data to inputs for model access
        for key, value in parenet_collated.items():
            if key not in ['ref_points', 'src_points', 'transform']:
                pylon_collated['inputs'][key] = value
        
        # Handle labels
        if 'transform' in parenet_collated:
            pylon_collated['labels']['transform'] = parenet_collated['transform']
        
        # Handle meta_info using parent class
        meta_info_list = [dp.get('meta_info', {}) for dp in datapoints]
        if any(meta_info_list):
            # Use parent class for meta_info collation
            meta_info_collated = super(PARENetCollator, self).__call__([{'meta_info': mi} for mi in meta_info_list])
            pylon_collated['meta_info'] = meta_info_collated.get('meta_info', {})
        
        return pylon_collated