import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from typing import Callable, Dict, List, Tuple, Any

class HungarianMatcher(nn.Module):
    """
    Generic Hungarian matcher that accepts custom cost functions.
    """
    def __init__(
        self, 
        cost_functions: Dict[str, Tuple[Callable, float]]
    ):
        """
        Initialize the matcher with custom cost functions.
        
        Args:
            cost_functions: Dictionary mapping cost names to (function, weight) tuples.
                            Each function should take (outputs, targets) and return a cost matrix.
        """
        super().__init__()
        self.cost_functions = cost_functions
        
    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform the matching using all registered cost functions.
        
        Args:
            outputs: Dict containing model outputs
            targets: Dict containing ground truth
            
        Returns:
            List of tuples (pred_idx, target_idx) for each batch element
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []
        
        # Iterate through batch size
        for b in range(bs):
            # Extract batch outputs and targets
            batch_outputs = {k: v[b] for k, v in outputs.items() if k != "aux_outputs"}
            batch_targets = targets[b]
            
            # Calculate all cost components
            cost_matrix = torch.zeros(
                (num_queries, len(batch_targets["labels"])), 
                device=batch_outputs["pred_logits"].device
            )
            
            # Apply each cost function
            for cost_name, (cost_fn, weight) in self.cost_functions.items():
                if weight != 0:
                    cost_component = cost_fn(batch_outputs, batch_targets)
                    cost_matrix += weight * cost_component
            
            # Use Hungarian algorithm for optimal assignment
            cost_matrix_cpu = cost_matrix.cpu().numpy()
            indices.append(linear_sum_assignment(cost_matrix_cpu))
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ] 