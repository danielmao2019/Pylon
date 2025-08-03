"""
Unit tests for PARENet collator integration.

Tests PARENet data collation functionality.
"""

import pytest
import torch
from data.collators.parenet.parenet_collator import PARENetCollator


def test_parenet_collator_instantiation():
    """Test that PARENet collator can be instantiated."""
    collator = PARENetCollator()
    assert collator is not None


def test_parenet_collator_call():
    """Test PARENet collator with properly formatted datapoints."""
    collator = PARENetCollator()
    
    # Create properly formatted datapoints following Pylon PCR dataset structure
    dummy_datapoints = [
        {
            'inputs': {
                'src_pc': torch.randn(100, 3),
                'tgt_pc': torch.randn(100, 3),
            },
            'labels': {
                'transform': torch.eye(4),
            },
            'meta_info': {
                'idx': 0,
                'dataset_name': 'test',
            }
        },
        {
            'inputs': {
                'src_pc': torch.randn(120, 3),
                'tgt_pc': torch.randn(120, 3),
            },
            'labels': {
                'transform': torch.eye(4),
            },
            'meta_info': {
                'idx': 1,
                'dataset_name': 'test',
            }
        }
    ]
    
    batch = collator(dummy_datapoints)
    
    # Verify batch structure matches PARENet expectations
    assert isinstance(batch, dict), "Batch must be a dictionary"
    
    # Check PARENet-specific output format
    expected_keys = ['points', 'lengths', 'features', 'batch_size']
    for key in expected_keys:
        assert key in batch, f"Missing required key: {key}"
    
    # Verify tensor shapes and types
    assert isinstance(batch['points'], torch.Tensor), "Points must be a tensor"
    assert isinstance(batch['lengths'], torch.Tensor), "Lengths must be a tensor"
    assert isinstance(batch['features'], torch.Tensor), "Features must be a tensor"
    assert isinstance(batch['batch_size'], int), "Batch size must be an integer"
    
    # Verify dimensions
    assert batch['points'].dim() == 2, "Points must be 2D tensor (N, 3)"
    assert batch['points'].shape[1] == 3, "Points must have 3 coordinates"
    assert batch['lengths'].dim() == 1, "Lengths must be 1D tensor"
    assert batch['features'].dim() == 2, "Features must be 2D tensor"
    assert batch['batch_size'] == 2, "Batch size should match input length"


def test_parenet_collator_empty_input():
    """Test PARENet collator with empty input."""
    collator = PARENetCollator()
    
    batch = collator([])
    
    # Verify empty batch structure
    assert isinstance(batch, dict), "Empty batch must still be a dictionary"
    assert batch['batch_size'] == 0, "Empty batch should have batch_size 0"
    
    # Check that tensor fields are empty but properly shaped
    if 'points' in batch:
        assert batch['points'].numel() == 0, "Empty batch should have empty points tensor"
    if 'lengths' in batch:
        assert batch['lengths'].numel() == 0, "Empty batch should have empty lengths tensor"
    if 'features' in batch:
        assert batch['features'].numel() == 0, "Empty batch should have empty features tensor"


def test_parenet_collator_single_datapoint():
    """Test PARENet collator with single datapoint."""
    collator = PARENetCollator()
    
    dummy_datapoint = {
        'inputs': {
            'src_pc': torch.randn(100, 3),
            'tgt_pc': torch.randn(100, 3),
        },
        'labels': {
            'transform': torch.eye(4),
        },
        'meta_info': {
            'idx': 0,
            'dataset_name': 'test',
        }
    }
    
    batch = collator([dummy_datapoint])
    
    # Verify single datapoint batch structure
    assert isinstance(batch, dict), "Single datapoint batch must be a dictionary"
    assert batch['batch_size'] == 1, "Single datapoint batch should have batch_size 1"
    
    # Verify tensor shapes for single datapoint
    assert batch['points'].shape[0] == 200, "Single pair should have 200 total points (100 + 100)"
    assert batch['lengths'].shape[0] == 2, "Single pair should have 2 length entries"
    assert torch.equal(batch['lengths'], torch.tensor([100, 100])), "Lengths should be [100, 100]"
