"""
Pylon-compatible wrapper for GMCNet model.

This module provides a Pylon-compatible API wrapper around the original GMCNet 
implementation while preserving the original code structure.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from easydict import EasyDict

from models.point_cloud_registration.gmcnet.gmcnet import Model as _GMCNetModel


class GMCNet(nn.Module):
    """Pylon-compatible wrapper for GMCNet model.
    
    This wrapper provides a Pylon-compatible API while preserving the original
    GMCNet implementation. The original model is accessed via self._model.
    """
    
    def __init__(self, args):
        super(GMCNet, self).__init__()
        
        # Handle both dictionary and object args using EasyDict
        if isinstance(args, dict):
            self.args = EasyDict(args)
        else:
            self.args = args
            
        self._model = _GMCNetModel(self.args)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass following Pylon API patterns.
        
        Args:
            inputs: Dictionary containing:
                - 'src_points': Source point cloud [B, N, 3]  
                - 'tgt_points': Target point cloud [B, N, 3]
                - 'transform': Ground truth transformation [B, 4, 4] (if available)
                - 'mode': 'train', 'val', or 'test' (optional, defaults to 'train')
                
        Returns:
            Dictionary containing model outputs
        """
        # Extract inputs and convert to original GMCNet format
        assert 'src_points' in inputs, "src_points must be provided in inputs"
        assert 'tgt_points' in inputs, "tgt_points must be provided in inputs"
        
        pts1 = inputs['src_points']
        pts2 = inputs['tgt_points']
        T_gt = inputs.get('transform', None)
        prefix = inputs.get('mode', 'train')
        
        # Call original GMCNet model
        if prefix == 'test':
            # Test mode: only transformation returned
            transformation = self._model(pts1, pts2, T_gt, prefix)
            return {
                'transformation': transformation,
                'T_12': transformation,  # Keep original key for compatibility
            }
        else:
            # Train/val mode: loss and metrics returned
            if T_gt is None:
                raise ValueError("Ground truth transformation must be provided during training/validation")
            
            loss, r_err, t_err, rmse, mse = self._model(pts1, pts2, T_gt, prefix)
            
            return {
                'loss': loss,
                'transformation': self._model.T_12,
                'T_12': self._model.T_12,  # Keep original key for compatibility
                'scores': self._model.scores12,
                'rotation_error': r_err,
                'translation_error': t_err,
                'rmse': rmse,
                'mse': mse,
                # Additional outputs for analysis
                'src_points': self._model.pts1,
                'tgt_points': self._model.pts2,
                'gt_transform': self._model.T_gt
            }
    
    def get_transform(self):
        """Get transformation results from the original model."""
        return self._model.get_transform()
    
    def visualize(self, i):
        """Visualization method from the original model."""
        return self._model.visualize(i)
