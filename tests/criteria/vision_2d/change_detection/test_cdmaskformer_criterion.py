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
    
    # Create change maps (binary segmentation masks)
    change_maps = torch.zeros((batch_size, height, width), dtype=torch.long)
    # First image has changes
    change_maps[0, 10:20, 10:20] = 1  # A square of class 1 in the middle
    # Second image has no changes
    
    # Create ground truth dictionary with change map
    y_true = {
        "change_map": change_maps
    }
    
    # Compute loss with change maps
    loss = criterion(outputs, y_true)
    
    # Basic checks on the loss
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    
    # Test with empty change maps (no changes)
    empty_change_maps = torch.zeros((batch_size, height, width), dtype=torch.long)
    empty_y_true = {
        "change_map": empty_change_maps
    }
    
    # Compute loss with empty change maps
    empty_loss = criterion(outputs, empty_y_true)
    
    # Check that we get a valid loss with empty change maps
    assert isinstance(empty_loss, torch.Tensor), "Loss should be a tensor"
    assert empty_loss.ndim == 0, "Loss should be a scalar"
    assert not torch.isnan(empty_loss), "Loss should not be NaN"
    assert not torch.isinf(empty_loss), "Loss should not be Inf"
