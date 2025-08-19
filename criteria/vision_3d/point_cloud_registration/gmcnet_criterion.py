"""
GMCNet criterion for point cloud registration.

GMCNet computes its own loss internally, so this criterion extracts
the loss from the model's output dictionary.
"""

from criteria.base_criterion import BaseCriterion
from typing import Dict, Optional
import torch


class GMCNetCriterion(BaseCriterion):
    """Criterion for GMCNet that extracts loss from model outputs."""
    
    def __init__(self):
        super().__init__(use_buffer=True)
    
    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract loss from GMCNet outputs.
        
        Args:
            y_pred: GMCNet model outputs containing 'loss' key
            y_true: Target labels (not used since GMCNet computes loss internally)
            
        Returns:
            Loss tensor from GMCNet internal computation
        """
        assert isinstance(y_pred, dict), f"Expected dict outputs from GMCNet, got {type(y_pred)}"
        assert 'loss' in y_pred, f"GMCNet outputs must contain 'loss' key, got keys: {list(y_pred.keys())}"
        
        loss = y_pred['loss']
        assert isinstance(loss, torch.Tensor), f"Expected loss to be tensor, got {type(loss)}"
        assert loss.numel() == 1, f"Expected scalar loss, got shape {loss.shape}"
        
        self.add_to_buffer(loss)
        return loss
    
    def summarize(self, output_path: Optional[str] = None) -> torch.Tensor:
        """Summarize loss trajectory across all data points in buffer."""
        assert self.use_buffer and hasattr(self, 'buffer') and self.buffer is not None
        self._buffer_queue.join()  # Wait for all items to be processed
        assert self._buffer_queue.empty(), "Buffer queue is not empty when summarizing"
        assert len(self.buffer) != 0

        # Summarize losses
        result = torch.stack(self.buffer, dim=0)
        assert result.ndim == 1, f"{result.shape=}"

        # Save to disk if path provided
        if output_path is not None:
            from utils.input_checks import check_write_file
            check_write_file(path=output_path)
            torch.save(obj=result, f=output_path)

        return result