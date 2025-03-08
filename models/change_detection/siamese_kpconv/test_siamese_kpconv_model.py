"""
Tests for the SiameseKPConv model.

This module contains tests for verifying the functionality of the SiameseKPConv model
for 3D point cloud change detection.
"""
import pytest
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from models.change_detection.siamese_kpconv.siamese_kpconv_model import SiameseKPConv
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
def model_config():
    """Fixture that returns a standard model configuration for testing."""
    return {
        'in_channels': 3,
        'out_channels': 2,
        'point_influence': 0.1,
        'down_channels': [16, 32, 64],  # Smaller network for testing
        'up_channels': [64, 32, 16],
        'bn_momentum': 0.02,
        'dropout': 0.1
    }


@pytest.fixture
def dummy_dataset():
    """Fixture that creates a dummy dataset for testing."""
    return DummyPointCloudDataset(num_samples=5, num_points=100, feature_dim=3)


@pytest.fixture
def data_loader(dummy_dataset):
    """Fixture that creates a DataLoader with the SiameseKPConvCollator."""
    collator = SiameseKPConvCollator()
    return DataLoader(
        dummy_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0  # Use 0 for easier debugging
    )


def test_SiameseKPConv_model(model_config, data_loader):
    """Test that the SiameseKPConv model works correctly with the DataLoader."""
    model = SiameseKPConv(**model_config)
    model.eval()
    
    # Get a batch from the DataLoader
    batch = next(iter(data_loader))
    
    # Forward pass through the model
    with torch.no_grad():
        output = model(batch['inputs'])
    
    # Check output shape
    num_points = batch['inputs']['pc_1']['pos'].shape[0]
    assert output.shape == (num_points, model_config['out_channels'])
    
    # Check that output contains raw logits (not normalized)
    assert torch.max(output).item() > 1.0 or torch.min(output).item() < 0.0


def test_train_iter(model_config, data_loader):
    """Test a single training iteration with the model and DataLoader."""
    model = SiameseKPConv(**model_config)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Single training step
    model.train()
    batch = next(iter(data_loader))
    
    # Forward pass
    output = model(batch['inputs'])
    
    # Calculate loss
    loss = criterion(output, batch['labels']['change'])
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check that gradients are computed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradients for {name}"
