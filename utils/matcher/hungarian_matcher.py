import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from typing import Callable, Dict, List, Tuple, Any, Union

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
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform the matching using all registered cost functions.
        
        Args:
            outputs: Dict containing model outputs
            targets: Dict or List of dicts containing ground truth
            
        Returns:
            List of tuples (pred_idx, target_idx) for each batch element
        """
        # Handle single dictionary case - convert to list of dictionaries
        if isinstance(targets, dict):
            targets = [targets]
        
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []
        
        # Iterate through batch size
        for b in range(bs):
            # Handle case when batch index exceeds number of targets
            if b >= len(targets):
                device = outputs["pred_logits"].device
                indices.append((torch.tensor([], dtype=torch.int64, device=device),
                               torch.tensor([], dtype=torch.int64, device=device)))
                continue
                
            # Extract batch outputs and targets
            batch_outputs = {k: v[b] for k, v in outputs.items() if k != "aux_outputs"}
            batch_targets = targets[b]
            
            if "labels" not in batch_targets:
                device = outputs["pred_logits"].device
                indices.append((torch.tensor([], dtype=torch.int64, device=device),
                               torch.tensor([], dtype=torch.int64, device=device)))
                continue
                
            if len(batch_targets["labels"]) == 0:
                device = outputs["pred_logits"].device
                indices.append((torch.tensor([], dtype=torch.int64, device=device),
                               torch.tensor([], dtype=torch.int64, device=device)))
                continue
                
            # Calculate all cost components
            num_labels = len(batch_targets["labels"])
            cost_matrix = torch.zeros(
                (num_queries, num_labels), 
                device=batch_outputs["pred_logits"].device
            )
            
            # Apply each cost function
            for cost_name, (cost_fn, weight) in self.cost_functions.items():
                if weight != 0:
                    try:
                        cost_component = cost_fn(batch_outputs, batch_targets)
                        if torch.isnan(cost_component).any():
                            cost_component = torch.nan_to_num(cost_component)
                        
                        cost_matrix += weight * cost_component
                    except Exception:
                        pass
            
            # Use Hungarian algorithm for optimal assignment
            try:
                cost_matrix_cpu = cost_matrix.cpu().numpy()
                i, j = linear_sum_assignment(cost_matrix_cpu)
                indices.append((torch.as_tensor(i, dtype=torch.int64, device=batch_outputs["pred_logits"].device),
                               torch.as_tensor(j, dtype=torch.int64, device=batch_outputs["pred_logits"].device)))
            except Exception:
                device = outputs["pred_logits"].device
                indices.append((torch.tensor([], dtype=torch.int64, device=device),
                               torch.tensor([], dtype=torch.int64, device=device)))
            
        return [
            (torch.as_tensor(i, dtype=torch.int64, device=outputs["pred_logits"].device),
             torch.as_tensor(j, dtype=torch.int64, device=outputs["pred_logits"].device))
            for i, j in indices
        ]
