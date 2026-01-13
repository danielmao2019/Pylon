"""
Unit tests for PARENet collator integration.

Tests PARENet data collation functionality.
"""

import torch

from data.collators.parenet.parenet_collator_wrapper import parenet_collate_fn
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def test_parenet_collate_fn_basic():
    """Test that PARENet collate function works with basic input."""
    # This test just verifies the function can be called
    assert callable(parenet_collate_fn)


def test_parenet_collator_call():
    """Test PARENet collator with properly formatted datapoints."""

    # Create properly formatted datapoints following Pylon PCR dataset structure
    dummy_datapoints = [
        {
            'inputs': {
                'src_pc': PointCloud(xyz=torch.randn(100, 3), data={'feat': torch.ones(100, 1)}),
                'tgt_pc': PointCloud(xyz=torch.randn(100, 3), data={'feat': torch.ones(100, 1)}),
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
                'src_pc': PointCloud(xyz=torch.randn(120, 3), data={'feat': torch.ones(120, 1)}),
                'tgt_pc': PointCloud(xyz=torch.randn(120, 3), data={'feat': torch.ones(120, 1)}),
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

    batch = parenet_collate_fn(
        dummy_datapoints,
        num_stages=4,
        voxel_size=0.3,
        num_neighbors=[16, 16, 16, 16],
        subsample_ratio=0.25
    )

    # Verify batch structure matches PARENet expectations
    assert isinstance(batch, dict), "Batch must be a dictionary"

    # Check Pylon structure with PARENet data inside inputs
    assert 'inputs' in batch, "Batch must contain 'inputs'"
    assert 'labels' in batch, "Batch must contain 'labels'"
    assert 'meta_info' in batch, "Batch must contain 'meta_info'"

    # Check PARENet-specific output format inside inputs
    inputs = batch['inputs']
    expected_keys = ['points', 'lengths', 'features', 'batch_size']
    for key in expected_keys:
        assert key in inputs, f"Missing required key in inputs: {key}"

    # Verify tensor shapes and types
    assert isinstance(inputs['points'], list), "Points must be a list of tensors for stack mode"
    assert isinstance(inputs['lengths'], list), "Lengths must be a list of tensors for stack mode"
    assert isinstance(inputs['features'], torch.Tensor), "Features must be a tensor"
    assert isinstance(inputs['batch_size'], int), "Batch size must be an integer"

    # Verify hierarchical structure for stack mode (points and lengths have 4 levels)
    assert len(inputs['points']) == 4, "Stack mode should have 4 hierarchical levels"
    assert len(inputs['lengths']) == 4, "Stack mode should have 4 hierarchical levels"
    assert inputs['batch_size'] == 2, "Batch size should match input length"

    # Verify feature tensor shape
    assert inputs['features'].dim() == 2, "Features must be 2D tensor"
    assert inputs['features'].shape[1] == 1, "Features should have 1 feature dimension"


def test_parenet_collator_single_datapoint():
    """Test PARENet collator with single datapoint."""

    dummy_datapoint = {
        'inputs': {
            'src_pc': PointCloud(xyz=torch.randn(100, 3), data={'feat': torch.ones(100, 1)}),
            'tgt_pc': PointCloud(xyz=torch.randn(100, 3), data={'feat': torch.ones(100, 1)}),
        },
        'labels': {
            'transform': torch.eye(4),
        },
        'meta_info': {
            'idx': 0,
            'dataset_name': 'test',
        }
    }

    batch = parenet_collate_fn(
        [dummy_datapoint],
        num_stages=4,
        voxel_size=0.3,
        num_neighbors=[16, 16, 16, 16],
        subsample_ratio=0.25
    )

    # Verify single datapoint batch structure
    assert isinstance(batch, dict), "Single datapoint batch must be a dictionary"
    assert 'inputs' in batch, "Batch must contain 'inputs'"

    inputs = batch['inputs']
    assert inputs['batch_size'] == 1, "Single datapoint batch should have batch_size 1"

    # Verify hierarchical structure for single datapoint stack mode
    assert len(inputs['points']) == 4, "Stack mode should have 4 hierarchical levels"
    assert len(inputs['lengths']) == 4, "Stack mode should have 4 hierarchical levels"

    # Check that first level contains the raw points for both src and tgt
    first_level_points = inputs['points'][0]
    first_level_lengths = inputs['lengths'][0]
    assert first_level_points.shape[0] == 200, "First level should have 200 total points (100 + 100)"
    assert first_level_lengths.shape[0] == 2, "First level should have 2 length entries"
    assert torch.equal(first_level_lengths, torch.tensor([100, 100])), "Lengths should be [100, 100]"
