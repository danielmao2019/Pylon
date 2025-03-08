"""
Tests for the SiameseKPConvCollator.

This module contains tests for verifying the functionality of the SiameseKPConvCollator
for 3D point cloud change detection.
"""
import pytest
import torch
from torch.utils.data import Dataset

from data.collators.siamese_kpconv_collator import SiameseKPConvCollator


class DummyPointCloudDataset(Dataset):
    """
    A simple dataset that produces point clouds and change maps for testing.
    """
    def __init__(self, num_samples=5, num_points=100, feature_dim=3):
        """
        Initialize a dummy dataset.
        
        Args:
            num_samples: Number of samples in the dataset
            num_points: Number of points per point cloud
            feature_dim: Number of features per point (excluding position)
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.feature_dim = feature_dim
        
        # Set fixed seed for reproducibility
        torch.manual_seed(42)
        
        # Generate random point clouds
        self.point_clouds_1 = [torch.rand(num_points, 3 + feature_dim) for _ in range(num_samples)]
        self.point_clouds_2 = [torch.rand(num_points, 3 + feature_dim) for _ in range(num_samples)]
        
        # Generate random change maps (1 means change, 0 means no change)
        self.change_maps = [torch.randint(0, 2, (num_points,), dtype=torch.long) for _ in range(num_samples)]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sample with inputs, labels, and meta_info.
        
        Returns:
            Dictionary with:
                - inputs: Dict with 'pc_0' and 'pc_1' point clouds
                - labels: Dict with 'change_map' tensor
                - meta_info: Dict with metadata
        """
        return {
            "inputs": {
                "pc_0": self.point_clouds_1[idx],
                "pc_1": self.point_clouds_2[idx]
            },
            "labels": {
                "change_map": self.change_maps[idx]
            },
            "meta_info": {
                "sample_idx": idx,
                "filename": f"dummy_sample_{idx}.ply"
            }
        }


@pytest.fixture
def dummy_dataset():
    """Fixture that creates a dummy dataset for testing."""
    return DummyPointCloudDataset(num_samples=5, num_points=100, feature_dim=3)


def test_siamese_kpconv_collator(dummy_dataset):
    """Test the SiameseKPConvCollator's functionality."""
    # Get two samples from the dataset
    samples = [dummy_dataset[0], dummy_dataset[1]]
    
    # Apply the collator
    collator = SiameseKPConvCollator()
    batch = collator(samples)
    
    # Check batch structure
    assert set(batch.keys()) == {'inputs', 'labels', 'meta_info'}
    assert set(batch['inputs'].keys()) == {'pc_0', 'pc_1'}
    assert set(batch['labels'].keys()) == {'change'}
    
    # Check that the point clouds were batched correctly
    for pc_key in ['pc_0', 'pc_1']:
        assert set(batch['inputs'][pc_key].keys()) == {'pos', 'x', 'batch'}
        assert batch['inputs'][pc_key]['pos'].shape[0] == dummy_dataset.num_points * 2
        assert batch['inputs'][pc_key]['x'].shape[0] == dummy_dataset.num_points * 2
        assert batch['inputs'][pc_key]['batch'].shape[0] == dummy_dataset.num_points * 2
        
        # Check batch indices
        expected_batch = torch.cat([
            torch.zeros(dummy_dataset.num_points, dtype=torch.long),
            torch.ones(dummy_dataset.num_points, dtype=torch.long)
        ])
        assert torch.all(batch['inputs'][pc_key]['batch'] == expected_batch)
    
    # Check change map
    assert batch['labels']['change'].shape[0] == dummy_dataset.num_points * 2
    
    # Check consistency between batched pc_1 and change map
    assert batch['inputs']['pc_1']['pos'].shape[0] == batch['labels']['change'].shape[0]
    
    # Verify the outputs can be used for model input
    assert isinstance(batch['inputs'], dict)
    assert all(isinstance(v, dict) for v in batch['inputs'].values())
    assert all(isinstance(v, torch.Tensor) for pc in batch['inputs'].values() for v in pc.values())
