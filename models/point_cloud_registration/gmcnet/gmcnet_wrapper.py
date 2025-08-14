"""
Pylon-compatible wrapper for GMCNet model.

This module provides a Pylon-compatible API wrapper around the original GMCNet 
implementation while preserving the original code structure and handling device 
compatibility intelligently.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from easydict import EasyDict

from models.point_cloud_registration.gmcnet.gmcnet import Model as _GMCNetModel


class GMCNet(nn.Module):
    """Pylon-compatible wrapper for GMCNet model.
    
    This wrapper provides a Pylon-compatible API while preserving the original
    GMCNet implementation. The wrapper handles device compatibility intelligently
    since the original GMCNet has hardcoded .cuda() calls.
    
    Key features:
    - Automatic device detection and management
    - Input validation with clear error messages  
    - Preserves original GMCNet functionality exactly
    - Handles both CPU and CUDA inputs transparently
    - Ensures tensor contiguity for C++ extensions
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
        """Forward pass following Pylon API patterns with device compatibility.
        
        Args:
            inputs: Dictionary containing:
                - 'src_points': Source point cloud [B, N, 3]  
                - 'tgt_points': Target point cloud [B, N, 3]
                - 'transform': Ground truth transformation [B, 4, 4] (if available)
                - 'mode': 'train', 'val', or 'test' (optional, defaults to 'train')
                
        Returns:
            Dictionary containing model outputs
            
        Raises:
            AssertionError: If required inputs are missing or have wrong shapes
            ValueError: If ground truth transform is missing in training/validation
            RuntimeError: If CUDA is required but not available
        """
        # GMCNet expects ModelNet40Dataset structure: inputs contains 'src_pc' and 'tgt_pc' dictionaries with 'pos' key
        assert 'src_pc' in inputs, f"GMCNet requires 'src_pc' in inputs, got keys: {list(inputs.keys())}"
        assert 'tgt_pc' in inputs, f"GMCNet requires 'tgt_pc' in inputs, got keys: {list(inputs.keys())}"
        
        # Extract point positions from the nested dictionary structure
        assert isinstance(inputs['src_pc'], dict), f"src_pc must be a dictionary, got {type(inputs['src_pc'])}"
        assert isinstance(inputs['tgt_pc'], dict), f"tgt_pc must be a dictionary, got {type(inputs['tgt_pc'])}"
        assert 'pos' in inputs['src_pc'], f"src_pc must contain 'pos' key, got keys: {list(inputs['src_pc'].keys())}"
        assert 'pos' in inputs['tgt_pc'], f"tgt_pc must contain 'pos' key, got keys: {list(inputs['tgt_pc'].keys())}"
        
        pts1 = inputs['src_pc']['pos']
        pts2 = inputs['tgt_pc']['pos']
        T_gt = inputs['transform']

        # In Pylon, always use train/val modes to get metrics from GMCNet
        # Never use 'test' mode as it only returns transformation without metrics
        if self.training:
            prefix = 'train'
        else:
            prefix = 'val'  # Use validation mode to get metrics during evaluation

        # Validate tensor shapes
        assert len(pts1.shape) == 3, f"src_points must be 3D [B, N, 3], got shape {pts1.shape}"
        assert len(pts2.shape) == 3, f"tgt_points must be 3D [B, N, 3], got shape {pts2.shape}"
        assert pts1.shape[-1] == 3, f"src_points last dim must be 3, got {pts1.shape[-1]}"
        assert pts2.shape[-1] == 3, f"tgt_points last dim must be 3, got {pts2.shape[-1]}"
        
        if T_gt is not None:
            assert len(T_gt.shape) == 3, f"transform must be 3D [B, 4, 4], got shape {T_gt.shape}"
            assert T_gt.shape[-2:] == (4, 4), f"transform must be [B, 4, 4], got {T_gt.shape}"
        
        # Device compatibility handling - use model's device
        input_device = pts1.device
        model_device = next(self._model.parameters()).device
        
        # GMCNet has hardcoded .cuda() calls, so model must be on CUDA
        if model_device.type != 'cuda':
            self._model = self._model.cuda()
            model_device = next(self._model.parameters()).device
        
        # Move inputs to model device and ensure contiguity for C++ extensions
        pts1_cuda = pts1.to(model_device).contiguous()
        pts2_cuda = pts2.to(model_device).contiguous()
        T_gt_cuda = T_gt.to(model_device).contiguous() if T_gt is not None else None
        
        # Call original GMCNet model
        if prefix == 'test':
            # Test mode: only transformation returned
            transformation = self._model(pts1_cuda, pts2_cuda, T_gt_cuda, prefix)
            outputs = {
                'transformation': transformation,
                'T_12': transformation,  # Keep original key for compatibility
            }
        else:
            # Train/val mode: loss and metrics returned
            if T_gt_cuda is None:
                raise ValueError("Ground truth transformation must be provided during training/validation")
            
            loss, r_err, t_err, rmse, mse = self._model(pts1_cuda, pts2_cuda, T_gt_cuda, prefix)
            
            outputs = {
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
        
        # Move outputs back to original device if different
        if input_device != model_device:
            outputs_original_device = {}
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    outputs_original_device[key] = value.to(input_device)
                else:
                    outputs_original_device[key] = value
            outputs = outputs_original_device
        
        return outputs
    
    def get_transform(self):
        """Get transformation results from the original model."""
        result = self._model.get_transform()
        # Handle case where original method returns tuple - extract just the transformation
        if isinstance(result, tuple):
            return result[0]  # Return just the transformation tensor
        return result
