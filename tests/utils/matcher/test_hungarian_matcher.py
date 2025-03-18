import pytest
import torch
import torch.nn.functional as F
from utils.matcher import HungarianMatcher

# Define cost functions for testing
def compute_test_class_cost(outputs, targets):
    """Simplified class cost for testing"""
    out_prob = outputs["pred_logits"].softmax(-1)  # [num_queries, num_classes+1]
    tgt_ids = targets["labels"]
    
    # Handle empty labels case
    if len(tgt_ids) == 0:
        return torch.zeros(out_prob.shape[0], 0, device=out_prob.device)
    
    # Get negative score for matched class (minimize cost)
    cost_class = -out_prob[:, tgt_ids]
    return cost_class

def compute_test_mask_cost(outputs, targets):
    """Simplified mask cost for testing"""
    out_mask = outputs["pred_masks"].flatten(1)  # [num_queries, H*W]
    tgt_mask = targets["masks"].flatten(1)  # [num_targets, H*W]
    
    # Handle empty mask case
    if tgt_mask.shape[0] == 0:
        return torch.zeros(out_mask.shape[0], 0, device=out_mask.device)
    
    # Simple L1 distance for testing
    cost_mask = torch.abs(out_mask.sigmoid().unsqueeze(1) - tgt_mask.unsqueeze(0)).mean(2)
    return cost_mask


def test_hungarian_matcher_empty_targets():
    """Test matcher with empty targets"""
    # Create a matcher with cost functions
    cost_functions = {
        "cost_class": (compute_test_class_cost, 1.0),
        "cost_mask": (compute_test_mask_cost, 2.0),
    }
    matcher = HungarianMatcher(cost_functions=cost_functions)
    
    # Create sample outputs and empty targets
    batch_size = 2
    num_queries = 5
    num_classes = 1
    height, width = 16, 16
    
    # Create model outputs
    pred_logits = torch.randn(batch_size, num_queries, num_classes + 1)
    pred_masks = torch.randn(batch_size, num_queries, height, width)
    
    outputs = {
        "pred_logits": pred_logits,
        "pred_masks": pred_masks
    }
    
    # Create empty targets
    targets = [
        {"labels": torch.tensor([], dtype=torch.int64), 
         "masks": torch.zeros((0, height, width))}
        for _ in range(batch_size)
    ]
    
    # Run matcher
    indices = matcher(outputs, targets)
    
    # Check outputs
    assert len(indices) == batch_size
    for i, j in indices:
        assert i.shape == j.shape
        assert len(i) == 0  # Should have no matches


def test_hungarian_matcher_with_targets():
    """Test matcher with populated targets"""
    # Create a matcher with cost functions
    cost_functions = {
        "cost_class": (compute_test_class_cost, 1.0),
        "cost_mask": (compute_test_mask_cost, 2.0),
    }
    matcher = HungarianMatcher(cost_functions=cost_functions)
    
    # Create sample data
    batch_size = 2
    num_queries = 10
    num_classes = 1
    height, width = 16, 16
    
    # Create model outputs
    pred_logits = torch.randn(batch_size, num_queries, num_classes + 1)
    pred_masks = torch.randn(batch_size, num_queries, height, width)
    
    outputs = {
        "pred_logits": pred_logits,
        "pred_masks": pred_masks
    }
    
    # Create targets with instances
    targets = []
    for b in range(batch_size):
        num_instances = 2  # Two instances per image
        labels = torch.ones(num_instances, dtype=torch.int64)  # All instances are class 1
        
        # Create fake masks for each instance
        masks = torch.zeros(num_instances, height, width)
        masks[0, 3:8, 3:8] = 1.0  # First instance
        masks[1, 9:14, 9:14] = 1.0  # Second instance
        
        targets.append({"labels": labels, "masks": masks})
    
    # Run matcher
    indices = matcher(outputs, targets)
    
    # Check outputs
    assert len(indices) == batch_size
    for i, j in indices:
        assert i.shape == j.shape
        assert len(i) == 2  # Should have 2 matches (one for each instance)
        assert len(j) == 2
        # Check that indices are within bounds
        assert torch.all(i >= 0) and torch.all(i < num_queries)
        assert torch.all(j >= 0) and torch.all(j < 2)
        # Check that matches are unique
        assert len(i.unique()) == len(i)
        assert len(j.unique()) == len(j)


def test_hungarian_matcher_dict_input():
    """Test matcher with dictionary input instead of list"""
    # Create a matcher with cost functions
    cost_functions = {
        "cost_class": (compute_test_class_cost, 1.0),
        "cost_mask": (compute_test_mask_cost, 2.0),
    }
    matcher = HungarianMatcher(cost_functions=cost_functions)
    
    # Create sample data
    batch_size = 1  # Only used with dict input for single batch
    num_queries = 10
    num_classes = 1
    height, width = 16, 16
    
    # Create model outputs
    pred_logits = torch.randn(batch_size, num_queries, num_classes + 1)
    pred_masks = torch.randn(batch_size, num_queries, height, width)
    
    outputs = {
        "pred_logits": pred_logits,
        "pred_masks": pred_masks
    }
    
    # Create a single target dictionary
    labels = torch.tensor([1], dtype=torch.int64)  # One instance of class 1
    masks = torch.zeros(1, height, width)
    masks[0, 5:10, 5:10] = 1.0  # Create a simple square mask
    
    targets = {"labels": labels, "masks": masks}
    
    # Run matcher
    indices = matcher(outputs, targets)
    
    # Check outputs
    assert len(indices) == batch_size
    for i, j in indices:
        assert i.shape == j.shape
        assert len(i) == 1  # Should have 1 match
        assert len(j) == 1
        # Check that indices are within bounds
        assert torch.all(i >= 0) and torch.all(i < num_queries)
        assert torch.all(j >= 0) and torch.all(j < 1)


def test_hungarian_matcher_weight_zero():
    """Test matcher with some cost weights set to zero"""
    # Create a matcher with one cost function weight set to zero
    cost_functions = {
        "cost_class": (compute_test_class_cost, 0.0),  # Zero weight
        "cost_mask": (compute_test_mask_cost, 2.0),
    }
    matcher = HungarianMatcher(cost_functions=cost_functions)
    
    # Create sample data
    batch_size = 2
    num_queries = 10
    num_classes = 1
    height, width = 16, 16
    
    # Create model outputs
    pred_logits = torch.randn(batch_size, num_queries, num_classes + 1)
    pred_masks = torch.randn(batch_size, num_queries, height, width)
    
    outputs = {
        "pred_logits": pred_logits,
        "pred_masks": pred_masks
    }
    
    # Create targets with instances
    targets = []
    for b in range(batch_size):
        num_instances = 2  
        labels = torch.ones(num_instances, dtype=torch.int64)  # All instances are class 1
        
        # Create fake masks for each instance
        masks = torch.zeros(num_instances, height, width)
        masks[0, 3:8, 3:8] = 1.0  # First instance
        masks[1, 9:14, 9:14] = 1.0  # Second instance
        
        targets.append({"labels": labels, "masks": masks})
    
    # Run matcher
    indices = matcher(outputs, targets)
    
    # Check outputs - should still work even with one weight at zero
    assert len(indices) == batch_size
    for i, j in indices:
        assert i.shape == j.shape
        assert len(i) == 2  # Should have 2 matches


def test_hungarian_matcher_different_cost_functions():
    """Test matcher with different cost functions"""
    # First create a matcher with only class cost
    cost_functions_class_only = {
        "cost_class": (compute_test_class_cost, 1.0),
    }
    matcher_class = HungarianMatcher(cost_functions=cost_functions_class_only)
    
    # Then create a matcher with only mask cost
    cost_functions_mask_only = {
        "cost_mask": (compute_test_mask_cost, 1.0),
    }
    matcher_mask = HungarianMatcher(cost_functions=cost_functions_mask_only)
    
    # Create sample data
    batch_size = 1
    num_queries = 10
    num_classes = 1
    height, width = 16, 16
    
    # Create model outputs
    # Make a specific output where some queries have good class scores but bad masks
    # and others have good masks but bad class scores
    pred_logits = torch.zeros(batch_size, num_queries, num_classes + 1)
    pred_masks = torch.zeros(batch_size, num_queries, height, width)
    
    # First 5 queries: good class score for class 1, bad masks
    pred_logits[0, :5, 1] = 5.0  # High score for class 1
    pred_logits[0, :5, 0] = -5.0  # Low score for class 0 (no object)
    
    # Last 5 queries: bad class score, good masks for the instance
    pred_logits[0, 5:, 0] = 5.0  # High score for no object
    pred_logits[0, 5:, 1] = -5.0  # Low score for class 1
    pred_masks[0, 5:, 5:10, 5:10] = 5.0  # Good mask prediction
    
    outputs = {
        "pred_logits": pred_logits,
        "pred_masks": pred_masks
    }
    
    # Create target with one instance
    labels = torch.tensor([1], dtype=torch.int64)  # One instance of class 1
    masks = torch.zeros(1, height, width)
    masks[0, 5:10, 5:10] = 1.0  # Create a simple square mask
    
    targets = [{"labels": labels, "masks": masks}]
    
    # Run matchers
    indices_class = matcher_class(outputs, targets)
    indices_mask = matcher_mask(outputs, targets)
    
    # Class-only matcher should prefer the first 5 queries (with good class scores)
    i_class, j_class = indices_class[0]
    assert i_class.item() < 5
    
    # Mask-only matcher should prefer the last 5 queries (with good masks)
    i_mask, j_mask = indices_mask[0]
    assert i_mask.item() >= 5 


def test_hungarian_matcher_with_cdmaskformer_criterion():
    """Test the interaction between HungarianMatcher and CDMaskFormerCriterion"""
    pytest.skip("This test is for CDMaskFormerCriterion integration and should be moved to the appropriate test file")


def test_hungarian_matcher_direct_with_change_map():
    """Test matcher directly with converted change_map format"""
    # Create a matcher with cost functions
    cost_functions = {
        "cost_class": (compute_test_class_cost, 1.0),
        "cost_mask": (compute_test_mask_cost, 2.0),
    }
    matcher = HungarianMatcher(cost_functions=cost_functions)
    
    # Create sample data
    batch_size = 2
    num_queries = 10
    num_classes = 1
    height, width = 32, 32
    
    # Create model outputs
    pred_logits = torch.randn(batch_size, num_queries, num_classes + 1)
    pred_masks = torch.randn(batch_size, num_queries, height, width)
    
    outputs = {
        "pred_logits": pred_logits,
        "pred_masks": pred_masks
    }
    
    # Create change map (what causes the bug in production)
    change_map = torch.zeros(batch_size, height, width, dtype=torch.long)
    change_map[:, 10:20, 10:20] = 1  # Add some change regions
    
    # Convert change_map to expected format manually (what the criterion does)
    formatted_targets = []
    for b in range(batch_size):
        batch_mask = change_map[b]
        if torch.any(batch_mask == 1):
            # Create a binary mask for change
            binary_mask = (batch_mask == 1).float().unsqueeze(0)
            # Create labels tensor (1 for change class)
            labels = torch.tensor([1], dtype=torch.int64)
            formatted_targets.append({
                'labels': labels,
                'masks': binary_mask
            })
        else:
            # No change regions
            formatted_targets.append({
                'labels': torch.tensor([], dtype=torch.int64),
                'masks': torch.zeros((0, batch_mask.shape[0], batch_mask.shape[1]), dtype=torch.float)
            })
    
    # Run matcher
    indices = matcher(outputs, formatted_targets)
    
    # Check outputs
    assert len(indices) == batch_size
    for i, j in indices:
        assert i.shape == j.shape
        assert len(i) == 1  # Should have 1 match per batch (one change region)
        assert len(j) == 1
