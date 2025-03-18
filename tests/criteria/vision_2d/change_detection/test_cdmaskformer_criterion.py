import pytest
import torch
from criteria.vision_2d.change_detection import CDMaskFormerCriterion

def test_cdmaskformer_criterion() -> None:
    # Create a sample batch size and dimensions
    batch_size = 2
    num_queries = 5
    num_classes = 1
    height, width = 32, 32
    
    # Create the criterion
    criterion = CDMaskFormerCriterion(num_classes=num_classes)
    
    # Create fake model outputs (what would come from CDMaskFormer)
    pred_logits = torch.randn(batch_size, num_queries, num_classes + 1)  # Class logits (+1 for no-object)
    pred_masks = torch.randn(batch_size, num_queries, height // 4, width // 4)  # Mask logits at lower resolution
    
    outputs = {
        "pred_logits": pred_logits,
        "pred_masks": pred_masks
    }
    
    # Create formatted targets that match the expected format in the criterion
    targets = []
    for b in range(batch_size):
        # Create a single instance mask
        masks = torch.zeros((1, height, width), dtype=torch.float)
        masks[0, 10:20, 10:20] = 1.0  # A square in the middle
        
        # Create labels tensor (1 for change class)
        labels = torch.tensor([1], dtype=torch.int64)
        
        targets.append({
            'labels': labels,
            'masks': masks
        })
    
    # Compute loss
    loss = criterion(outputs, targets)
    
    # Basic checks on the loss
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    
    # Test with empty targets
    empty_targets = []
    for _ in range(batch_size):
        empty_targets.append({
            'labels': torch.tensor([], dtype=torch.int64),
            'masks': torch.zeros((0, height, width), dtype=torch.float)
        })
    
    # Compute loss with empty targets
    empty_loss = criterion(outputs, empty_targets)
    
    # Check that the loss is valid
    assert isinstance(empty_loss, torch.Tensor), "Loss should be a tensor"
    assert empty_loss.ndim == 0, "Loss should be a scalar"
    assert empty_loss.item() > 0, "Loss with empty targets should still be positive (classification error)"
