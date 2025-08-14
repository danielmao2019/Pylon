"""
GMCNet metric for point cloud registration.

GMCNet outputs multiple metrics internally. This metric extracts and tracks
the key registration metrics.
"""

import torch
from typing import Dict, Union
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
    
    def __call__(self, datapoint: Dict[str, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]) -> Dict[str, torch.Tensor]:
        """Override __call__ to handle GMCNet's multiple output keys.
        
        GMCNet outputs multiple keys (loss, T_12, scores, rotation_error, etc.) but 
        SingleTaskMetric expects exactly one key. We need to extract the relevant
        metrics from the GMCNet outputs.
        """
        # Extract outputs and labels from datapoint
        assert 'outputs' in datapoint and 'labels' in datapoint
        y_pred = datapoint['outputs']
        y_true = datapoint['labels']
        
        # GMCNet outputs a dict with multiple metrics - extract them directly
        assert isinstance(y_pred, dict), f"Expected dict outputs from GMCNet, got {type(y_pred)}"
        
        metrics = {}
        
        # Extract available metrics from GMCNet outputs
        if 'loss' in y_pred:
            metrics['loss'] = y_pred['loss']
        if 'rmse' in y_pred:
            metrics['rmse'] = y_pred['rmse']
        if 'mse' in y_pred:
            metrics['mse'] = y_pred['mse']
        if 'rotation_error' in y_pred:
            metrics['rotation_error'] = y_pred['rotation_error']
        if 'translation_error' in y_pred:
            metrics['translation_error'] = y_pred['translation_error']
        
        # Ensure we have at least some metrics
        assert len(metrics) > 0, f"No metrics found in GMCNet outputs: {list(y_pred.keys())}"
        
        # Add to buffer for aggregation
        self.add_to_buffer(metrics, datapoint)
        return metrics
