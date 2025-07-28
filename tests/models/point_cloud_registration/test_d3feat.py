"""Tests for D3Feat model integration."""

from typing import Dict, Any, Tuple
import pytest
import torch
import numpy as np

from models.point_cloud_registration.d3feat import D3FeatModel
from data.datasets.base_dataset import BaseDataset
from utils.builders.builder import build_from_config


class DummyD3FeatDataset(BaseDataset):
    """Dummy dataset for testing D3Feat with simple point cloud data."""
    
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {'train': 10, 'val': 5, 'test': 5}
    INPUT_NAMES = ['src_pc', 'tgt_pc', 'correspondences']
    LABEL_NAMES = ['transform']
    SHA1SUM = None
    
    def __init__(
        self,
        num_points_src: int = 512,
        num_points_tgt: int = 512,
        num_correspondences: int = 50,
        **kwargs
    ) -> None:
        """Initialize dummy dataset.
        
        Args:
            num_points_src: Number of points in source cloud
            num_points_tgt: Number of points in target cloud
            num_correspondences: Number of correspondences
        """
        self.num_points_src = num_points_src
        self.num_points_tgt = num_points_tgt
        self.num_correspondences = num_correspondences
        super(DummyD3FeatDataset, self).__init__(**kwargs)
        
    def _init_annotations(self) -> None:
        """Initialize dummy annotations."""
        self.annotations = list(range(self.DATASET_SIZE[self.split]))
        
    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load dummy datapoint with random data."""
        # Generate random point clouds
        src_pos = torch.randn(self.num_points_src, 3, dtype=torch.float32)
        tgt_pos = torch.randn(self.num_points_tgt, 3, dtype=torch.float32)
        
        # Simple features (ones for D3Feat)
        src_feat = torch.ones(self.num_points_src, 1, dtype=torch.float32)
        tgt_feat = torch.ones(self.num_points_tgt, 1, dtype=torch.float32)
        
        # Random correspondences
        corr_src = torch.randint(0, self.num_points_src, (self.num_correspondences,))
        corr_tgt = torch.randint(0, self.num_points_tgt, (self.num_correspondences,))
        correspondences = torch.stack([corr_src, corr_tgt], dim=1).long()
        
        # Random transform
        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, :3] = torch.randn(3, 3, dtype=torch.float32)
        transform[:3, 3] = torch.randn(3, dtype=torch.float32)
        
        inputs = {
            'src_pc': {'pos': src_pos, 'feat': src_feat},
            'tgt_pc': {'pos': tgt_pos, 'feat': tgt_feat},
            'correspondences': correspondences,
        }
        
        labels = {
            'transform': transform,
        }
        
        meta_info = {
            'num_src_points': self.num_points_src,
            'num_tgt_points': self.num_points_tgt,
        }
        
        return inputs, labels, meta_info


def test_d3feat_model_initialization():
    """Test D3Feat model initialization."""
    # Test with default parameters
    model = D3FeatModel()
    assert model is not None
    assert hasattr(model, 'd3feat_model')
    assert hasattr(model, 'config')
    
    # Test with custom parameters
    model = D3FeatModel(
        num_layers=3,
        first_features_dim=64,
        conv_radius=2.0,
    )
    assert model.config.num_layers == 3
    assert model.config.first_features_dim == 64
    assert model.config.conv_radius == 2.0


def test_d3feat_model_forward():
    """Test D3Feat model forward pass."""
    # Initialize model
    model = D3FeatModel(num_layers=3)  # Smaller for testing
    model.eval()
    
    # Create dummy input
    batch_size = 1
    num_points_src = 256
    num_points_tgt = 256
    
    src_pos = torch.randn(num_points_src, 3, dtype=torch.float32)
    tgt_pos = torch.randn(num_points_tgt, 3, dtype=torch.float32)
    src_feat = torch.ones(num_points_src, 1, dtype=torch.float32)
    tgt_feat = torch.ones(num_points_tgt, 1, dtype=torch.float32)
    
    inputs = {
        'src_pc': {'pos': src_pos, 'feat': src_feat},
        'tgt_pc': {'pos': tgt_pos, 'feat': tgt_feat},
        'correspondences': torch.zeros(0, 2, dtype=torch.long),  # Empty for inference
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)
    
    # Check outputs
    assert 'descriptors' in outputs
    assert 'scores' in outputs
    
    # Check shapes
    total_points = num_points_src + num_points_tgt
    assert outputs['descriptors'].shape[0] == total_points
    assert outputs['descriptors'].shape[1] == 32  # Default feature dimension
    assert outputs['scores'].shape[0] == total_points
    assert outputs['scores'].shape[1] == 1
    
    # Check normalization
    desc_norms = torch.norm(outputs['descriptors'], p=2, dim=1)
    assert torch.allclose(desc_norms, torch.ones_like(desc_norms), atol=1e-5)


def test_d3feat_model_with_correspondences():
    """Test D3Feat model with correspondences."""
    model = D3FeatModel(num_layers=3)
    model.eval()
    
    # Create input with correspondences
    num_points = 256
    num_corr = 20
    
    src_pos = torch.randn(num_points, 3, dtype=torch.float32)
    tgt_pos = torch.randn(num_points, 3, dtype=torch.float32)
    src_feat = torch.ones(num_points, 1, dtype=torch.float32)
    tgt_feat = torch.ones(num_points, 1, dtype=torch.float32)
    
    # Create valid correspondences
    corr_idx = torch.randint(0, num_points, (num_corr, 2), dtype=torch.long)
    
    inputs = {
        'src_pc': {'pos': src_pos, 'feat': src_feat},
        'tgt_pc': {'pos': tgt_pos, 'feat': tgt_feat},
        'correspondences': corr_idx,
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)
    
    # Verify outputs
    assert outputs['descriptors'].shape[0] == 2 * num_points
    assert outputs['scores'].shape[0] == 2 * num_points


def test_d3feat_gradient_flow():
    """Test gradient flow through D3Feat model."""
    model = D3FeatModel(num_layers=3)
    model.train()
    
    # Create input
    num_points = 128
    src_pos = torch.randn(num_points, 3, dtype=torch.float32, requires_grad=True)
    tgt_pos = torch.randn(num_points, 3, dtype=torch.float32, requires_grad=True)
    src_feat = torch.ones(num_points, 1, dtype=torch.float32)
    tgt_feat = torch.ones(num_points, 1, dtype=torch.float32)
    
    inputs = {
        'src_pc': {'pos': src_pos, 'feat': src_feat},
        'tgt_pc': {'pos': tgt_pos, 'feat': tgt_feat},
        'correspondences': torch.zeros(0, 2, dtype=torch.long),
    }
    
    # Forward pass
    outputs = model(inputs)
    
    # Create dummy loss
    loss = outputs['descriptors'].sum() + outputs['scores'].sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_d3feat_with_dataset():
    """Test D3Feat with dummy dataset."""
    # Create dataset
    dataset = DummyD3FeatDataset(
        num_points_src=256,
        num_points_tgt=256,
        num_correspondences=30,
        split='train'
    )
    
    # Initialize model
    model = D3FeatModel(num_layers=3)
    model.eval()
    
    # Get a datapoint and ensure all tensors are on CPU
    datapoint = dataset[0]
    inputs = datapoint['inputs']
    
    # Move everything to CPU to avoid device issues in testing
    for key in inputs:
        if isinstance(inputs[key], dict):
            for subkey in inputs[key]:
                if isinstance(inputs[key][subkey], torch.Tensor):
                    inputs[key][subkey] = inputs[key][subkey].cpu()
        elif isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].cpu()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)
    
    # Verify outputs
    assert 'descriptors' in outputs
    assert 'scores' in outputs
    assert outputs['descriptors'].shape[0] == 512  # src + tgt
    assert outputs['scores'].shape[0] == 512


def test_d3feat_model_config():
    """Test D3Feat model configuration building."""
    config = {
        'class': D3FeatModel,
        'args': {
            'num_layers': 4,
            'first_features_dim': 96,
            'conv_radius': 3.0,
            'num_kernel_points': 20,
        }
    }
    
    # Build model from config
    model = build_from_config(config)
    
    assert isinstance(model, D3FeatModel)
    assert model.config.num_layers == 4
    assert model.config.first_features_dim == 96
    assert model.config.conv_radius == 3.0
    assert model.config.num_kernel_points == 20


if __name__ == '__main__':
    # Run tests
    test_d3feat_model_initialization()
    print("✓ Model initialization test passed")
    
    test_d3feat_model_forward()
    print("✓ Model forward pass test passed")
    
    test_d3feat_model_with_correspondences()
    print("✓ Model with correspondences test passed")
    
    test_d3feat_gradient_flow()
    print("✓ Gradient flow test passed")
    
    test_d3feat_with_dataset()
    print("✓ Dataset integration test passed")
    
    test_d3feat_model_config()
    print("✓ Model config test passed")
    
    print("\nAll D3Feat model tests passed!")