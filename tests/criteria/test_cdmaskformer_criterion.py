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
    criterion = CDMaskFormerCriterion(num_classes=num_classes, ignore_value=255)
    
    # Create fake model outputs (what would come from CDMaskFormer)
    pred_logits = torch.randn(batch_size, num_queries, num_classes + 1)  # Class logits (+1 for no-object)
    pred_masks = torch.randn(batch_size, num_queries, height, width)  # Mask logits
    
    outputs = {
        "pred_logits": pred_logits,
        "pred_masks": pred_masks
    }
    
    # Create fake target with ground truth labels
    # Values: 0 for no change, 1 for change, 255 for ignore
    labels = torch.zeros(batch_size, height, width, dtype=torch.long)
    
    # Add some change regions
    labels[:, 10:20, 10:20] = 1
    
    # Add some ignore regions
    labels[:, 25:30, 25:30] = 255
    
    targets = {
        "labels": labels
    }
    
    # Compute loss
    loss = criterion(outputs, targets)
    
    # Basic checks on the loss
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    
    # Test with all pixels ignored
    all_ignored_labels = torch.full((batch_size, height, width), 255, dtype=torch.long)
    all_ignored_targets = {"labels": all_ignored_labels}
    
    # This should handle gracefully and not raise an error
    all_ignored_loss = criterion(outputs, all_ignored_targets)
    
    # The loss should be zero since no valid pixels to compute on
    assert all_ignored_loss.item() == 0, "Loss with all pixels ignored should be zero"
