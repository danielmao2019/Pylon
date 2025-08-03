"""
GMCNet metric for point cloud registration.

GMCNet outputs multiple metrics internally. This metric extracts and tracks
the key registration metrics.
"""

import torch
from typing import Dict
from metrics.wrappers.single_task_metric import SingleTaskMetric


class GMCNetMetric(SingleTaskMetric):
    """Metric extractor for GMCNet that uses internal model outputs."""
    
    # Required for early stopping and model comparison
    DIRECTIONS = {
        'loss': -1,           # Lower loss is better
        'rmse': -1,           # Lower RMSE is better
        'mse': -1,            # Lower MSE is better  
        'rotation_error': -1, # Lower rotation error is better
        'translation_error': -1, # Lower translation error is better
    }
    
    def __init__(self):
        super().__init__(use_buffer=True)
    
    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract metrics from GMCNet outputs.
        
        For GMCNet, y_pred should be a dictionary containing the computed metrics.
        Since SingleTaskMetric expects tensors, this is a bit of an adaptation.
        The training loop will need to handle this specially.
        
        Args:
            y_pred: GMCNet model outputs (expects dict but will be adapted)
            y_true: Target labels (not used since GMCNet computes metrics internally)
            
        Returns:
            Dictionary of metrics extracted from model outputs
        """
        # For GMCNet, we expect the training loop to pass the entire outputs dict
        # as a single "tensor" (actually dict) since GMCNet computes metrics internally
        
        if hasattr(y_pred, 'keys'):  # It's a dict (GMCNet outputs)
            outputs = y_pred
            metrics = {}
            
            # Extract available metrics from GMCNet outputs
            if 'loss' in outputs:
                metrics['loss'] = outputs['loss']
            if 'rmse' in outputs:
                metrics['rmse'] = outputs['rmse']
            if 'mse' in outputs:
                metrics['mse'] = outputs['mse']
            if 'rotation_error' in outputs:
                metrics['rotation_error'] = outputs['rotation_error']
            if 'translation_error' in outputs:
                metrics['translation_error'] = outputs['translation_error']
                
            return metrics
        else:
            # Fallback for single tensor input
            return {'score': y_pred}
