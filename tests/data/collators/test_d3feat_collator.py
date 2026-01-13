"""Tests for D3Feat collator."""

from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from data.collators.d3feat import D3FeatCollator
from data.datasets.base_dataset import BaseDataset
from data.structures.three_d.point_cloud.point_cloud import PointCloud


class SimplePCRDataset(BaseDataset):
    """Simple dataset for testing D3Feat collator."""

    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {'train': 5, 'val': 5, 'test': 5}
    INPUT_NAMES = ['src_pc', 'tgt_pc', 'correspondences']
    LABEL_NAMES = ['transform']
    SHA1SUM = None

    def __init__(self, num_points: int = 128, **kwargs):
        self.num_points = num_points
        super(SimplePCRDataset, self).__init__(**kwargs)

    def _init_annotations(self):
        """Initialize simple annotations."""
        self.annotations = list(range(self.DATASET_SIZE[self.split]))

    def _load_datapoint(self, idx: int):
        """Load simple test datapoint."""
        # Generate random point clouds
        src_pos = torch.randn(self.num_points, 3, dtype=torch.float32)
        tgt_pos = torch.randn(self.num_points, 3, dtype=torch.float32)
        src_feat = torch.ones(self.num_points, 1, dtype=torch.float32)
        tgt_feat = torch.ones(self.num_points, 1, dtype=torch.float32)

        # Random correspondences
        num_corr = min(20, self.num_points)
        corr_src = torch.randint(0, self.num_points, (num_corr,))
        corr_tgt = torch.randint(0, self.num_points, (num_corr,))
        correspondences = torch.stack([corr_src, corr_tgt], dim=1).long()

        # Random transform
        transform = torch.eye(4, dtype=torch.float32)

        inputs = {
            'src_pc': PointCloud(xyz=src_pos, data={'feat': src_feat}),
            'tgt_pc': PointCloud(xyz=tgt_pos, data={'feat': tgt_feat}),
            'correspondences': correspondences,
        }

        labels = {
            'transform': transform,
        }

        meta_info = {
            'num_src_points': self.num_points,
            'num_tgt_points': self.num_points,
        }

        return inputs, labels, meta_info


def test_d3feat_collator_initialization():
    """Test D3FeatCollator initialization."""
    # Default initialization
    collator = D3FeatCollator()
    assert collator is not None
    assert hasattr(collator, 'num_layers')
    assert hasattr(collator, 'architecture')

    # Custom initialization
    collator = D3FeatCollator(
        num_layers=3,
        first_subsampling_dl=0.05,
        conv_radius=3.0,
        neighborhood_limits=[15, 15, 15]
    )
    assert collator.num_layers == 3
    assert collator.first_subsampling_dl == 0.05
    assert collator.conv_radius == 3.0
    assert collator.neighborhood_limits == [15, 15, 15]


def test_d3feat_collator_single_item():
    """Test D3FeatCollator with single item."""
    collator = D3FeatCollator(num_layers=3)

    # Create dummy datapoint
    src_pos = torch.randn(64, 3, dtype=torch.float32)
    tgt_pos = torch.randn(64, 3, dtype=torch.float32)
    src_feat = torch.ones(64, 1, dtype=torch.float32)
    tgt_feat = torch.ones(64, 1, dtype=torch.float32)
    correspondences = torch.randint(0, 64, (10, 2), dtype=torch.long)

    datapoint = {
        'inputs': {
            'src_pc': PointCloud(xyz=src_pos, data={'feat': src_feat}),
            'tgt_pc': PointCloud(xyz=tgt_pos, data={'feat': tgt_feat}),
            'correspondences': correspondences,
        },
        'labels': {
            'transform': torch.eye(4, dtype=torch.float32),
        },
        'meta_info': {
            'idx': 0,
        }
    }

    # Collate single item
    batch = collator([datapoint])

    # Check required fields
    assert 'points' in batch
    assert 'neighbors' in batch
    assert 'pools' in batch
    assert 'upsamples' in batch
    assert 'features' in batch
    assert 'stack_lengths' in batch
    assert 'corr' in batch
    assert 'dist_keypts' in batch
    assert 'pylon_batch' in batch

    # Check types
    assert isinstance(batch['points'], list)
    assert isinstance(batch['neighbors'], list)
    assert isinstance(batch['features'], torch.Tensor)
    assert isinstance(batch['corr'], torch.Tensor)

    # Check original batch is preserved
    assert len(batch['pylon_batch']) == 1
    assert batch['pylon_batch'][0] == datapoint


def test_d3feat_collator_multiple_items():
    """Test D3FeatCollator with multiple items."""
    collator = D3FeatCollator(num_layers=3)

    # Create multiple datapoints
    batch_data = []
    for i in range(2):
        src_pos = torch.randn(32, 3, dtype=torch.float32)
        tgt_pos = torch.randn(32, 3, dtype=torch.float32)
        src_feat = torch.ones(32, 1, dtype=torch.float32)
        tgt_feat = torch.ones(32, 1, dtype=torch.float32)
        correspondences = torch.randint(0, 32, (5, 2), dtype=torch.long)

        datapoint = {
            'inputs': {
                'src_pc': PointCloud(xyz=src_pos, data={'feat': src_feat}),
                'tgt_pc': PointCloud(xyz=tgt_pos, data={'feat': tgt_feat}),
                'correspondences': correspondences,
            },
            'labels': {
                'transform': torch.eye(4, dtype=torch.float32),
            },
            'meta_info': {
                'idx': i,
            }
        }
        batch_data.append(datapoint)

    # Collate batch
    batch = collator(batch_data)

    # Check batch processing
    assert len(batch['pylon_batch']) == 2

    # Features should be concatenated
    # Each datapoint has 32+32=64 points, 2 datapoints = 128 total
    expected_total_points = 2 * (32 + 32)  # 2 batches * (src + tgt points)
    assert batch['features'].shape[0] == expected_total_points


def test_d3feat_collator_empty_correspondences():
    """Test D3FeatCollator with empty correspondences."""
    collator = D3FeatCollator(num_layers=3)

    # Create datapoint with no correspondences
    src_pos = torch.randn(50, 3, dtype=torch.float32)
    tgt_pos = torch.randn(50, 3, dtype=torch.float32)
    src_feat = torch.ones(50, 1, dtype=torch.float32)
    tgt_feat = torch.ones(50, 1, dtype=torch.float32)

    datapoint = {
        'inputs': {
            'src_pc': PointCloud(xyz=src_pos, data={'feat': src_feat}),
            'tgt_pc': PointCloud(xyz=tgt_pos, data={'feat': tgt_feat}),
            'correspondences': torch.zeros(0, 2, dtype=torch.long),
        },
        'labels': {
            'transform': torch.eye(4, dtype=torch.float32),
        },
        'meta_info': {
            'idx': 0,
        }
    }

    # Should handle empty correspondences gracefully
    batch = collator([datapoint])

    assert batch['corr'].shape[0] == 0
    assert batch['dist_keypts'].shape == (0, 0)


def test_d3feat_collator_with_dataset():
    """Test D3FeatCollator with actual dataset."""
    dataset = SimplePCRDataset(num_points=64, split='train')
    collator = D3FeatCollator(num_layers=3)

    # Get datapoints from dataset
    datapoints = [dataset[i] for i in range(2)]

    # Collate
    batch = collator(datapoints)

    # Verify structure
    assert 'points' in batch
    assert 'features' in batch
    assert len(batch['pylon_batch']) == 2

    # Check that points and features match expected sizes
    total_points = 2 * (64 + 64)  # 2 batches * (src + tgt)
    assert batch['features'].shape[0] == total_points


def test_d3feat_collator_tensor_types():
    """Test D3FeatCollator produces correct tensor types."""
    collator = D3FeatCollator(num_layers=3)

    # Create datapoint
    src_pos = torch.randn(32, 3, dtype=torch.float32)
    tgt_pos = torch.randn(32, 3, dtype=torch.float32)
    src_feat = torch.ones(32, 1, dtype=torch.float32)
    tgt_feat = torch.ones(32, 1, dtype=torch.float32)
    correspondences = torch.randint(0, 32, (8, 2), dtype=torch.long)

    datapoint = {
        'inputs': {
            'src_pc': PointCloud(xyz=src_pos, data={'feat': src_feat}),
            'tgt_pc': PointCloud(xyz=tgt_pos, data={'feat': tgt_feat}),
            'correspondences': correspondences,
        },
        'labels': {
            'transform': torch.eye(4, dtype=torch.float32),
        },
        'meta_info': {'idx': 0}
    }

    batch = collator([datapoint])

    # Check tensor dtypes
    assert batch['features'].dtype == torch.float32
    for points in batch['points']:
        assert points.dtype == torch.float32
    for neighbors in batch['neighbors']:
        assert neighbors.dtype == torch.int64
    assert batch['corr'].dtype == torch.int64


if __name__ == '__main__':
    # Run tests
    test_d3feat_collator_initialization()
    print("✓ D3FeatCollator initialization test passed")

    test_d3feat_collator_single_item()
    print("✓ D3FeatCollator single item test passed")

    test_d3feat_collator_multiple_items()
    print("✓ D3FeatCollator multiple items test passed")

    test_d3feat_collator_empty_correspondences()
    print("✓ D3FeatCollator empty correspondences test passed")

    test_d3feat_collator_with_dataset()
    print("✓ D3FeatCollator with dataset test passed")

    test_d3feat_collator_tensor_types()
    print("✓ D3FeatCollator tensor types test passed")

    print("\nAll D3Feat collator tests passed!")
