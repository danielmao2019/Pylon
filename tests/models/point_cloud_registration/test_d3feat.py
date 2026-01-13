"""Tests for D3Feat model integration."""

from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from data.dataloaders.d3feat_dataloader import D3FeatDataLoader
from data.datasets.base_dataset import BaseDataset
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from models.point_cloud_registration.d3feat import D3FeatModel
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

    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, PointCloud], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load dummy datapoint with random data."""
        # Generate random point clouds (on CPU as per BaseDataset design)
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
            'src_pc': PointCloud(xyz=src_pos, data={'feat': src_feat}),
            'tgt_pc': PointCloud(xyz=tgt_pos, data={'feat': tgt_feat}),
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


def get_batched_input(model_config, num_points_src=256, num_points_tgt=256, num_correspondences=30, device=None):
    """Helper function to get properly batched D3Feat input data.

    Args:
        model_config: D3Feat model configuration
        num_points_src: Number of source points
        num_points_tgt: Number of target points
        num_correspondences: Number of correspondences
        device: Device to place data on (defaults to CUDA if available)

    Returns:
        Batched input ready for D3Feat model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy dataset with explicit device
    dataset = DummyD3FeatDataset(
        num_points_src=num_points_src,
        num_points_tgt=num_points_tgt,
        num_correspondences=num_correspondences,
        split='train',
        device=device  # Explicitly set device to ensure correct placement
    )

    # Create dataloader with model config
    dataloader = D3FeatDataLoader(
        dataset=dataset,
        config=model_config,
        batch_size=1,
        shuffle=False,
        num_workers=0  # Single-threaded for testing
    )

    # Get one batch
    for batch in dataloader:
        return batch


@pytest.fixture
def device():
    """Get appropriate compute device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def test_d3feat_model_forward(device):
    """Test D3Feat model forward pass."""
    # Initialize model on appropriate device
    model = D3FeatModel(num_layers=5)
    model = model.to(device)
    model.eval()

    # Get batched input using helper function
    batch = get_batched_input(model.config, num_points_src=256, num_points_tgt=256, num_correspondences=0, device=device)

    # Forward pass
    with torch.no_grad():
        outputs = model(batch['inputs'])

    # Check outputs
    assert 'descriptors' in outputs
    assert 'scores' in outputs

    # Check shapes
    total_points = 256 + 256
    assert outputs['descriptors'].shape[0] == total_points
    assert outputs['descriptors'].shape[1] == 32  # Default feature dimension
    assert outputs['scores'].shape[0] == total_points
    assert outputs['scores'].shape[1] == 1

    # Check normalization
    desc_norms = torch.norm(outputs['descriptors'], p=2, dim=1)
    assert torch.allclose(desc_norms, torch.ones_like(desc_norms), atol=1e-5)

    # Verify outputs are on correct device
    assert outputs['descriptors'].device.type == device.type
    assert outputs['scores'].device.type == device.type


def test_d3feat_model_with_correspondences(device):
    """Test D3Feat model with correspondences."""
    model = D3FeatModel(num_layers=5)
    model = model.to(device)
    model.eval()

    # Get batched input with correspondences
    batch = get_batched_input(model.config, num_points_src=256, num_points_tgt=256, num_correspondences=20, device=device)

    # Forward pass
    with torch.no_grad():
        outputs = model(batch['inputs'])

    # Verify outputs
    assert outputs['descriptors'].shape[0] == 2 * 256
    assert outputs['scores'].shape[0] == 2 * 256

    # Verify device placement
    assert outputs['descriptors'].device.type == device.type
    assert outputs['scores'].device.type == device.type


def test_d3feat_gradient_flow(device):
    """Test gradient flow through D3Feat model."""
    model = D3FeatModel(num_layers=5)
    model = model.to(device)
    model.train()

    # Get batched input
    batch = get_batched_input(model.config, num_points_src=128, num_points_tgt=128, num_correspondences=0, device=device)

    # Forward pass
    outputs = model(batch['inputs'])

    # Create dummy loss
    loss = outputs['descriptors'].sum() + outputs['scores'].sum()

    # Backward pass
    loss.backward()

    # Check gradients exist and are valid
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            # Verify gradients are on correct device
            assert param.grad.device.type == device.type, f"Gradient device mismatch for {name}"


def test_d3feat_with_dataset(device):
    """Test D3Feat with dummy dataset."""
    # Initialize model
    model = D3FeatModel(num_layers=5)
    model = model.to(device)
    model.eval()

    # Get batched input using dataset
    batch = get_batched_input(model.config, num_points_src=256, num_points_tgt=256, num_correspondences=30, device=device)

    # Forward pass
    with torch.no_grad():
        outputs = model(batch['inputs'])

    # Verify outputs
    assert 'descriptors' in outputs
    assert 'scores' in outputs
    assert outputs['descriptors'].shape[0] == 512  # src + tgt
    assert outputs['scores'].shape[0] == 512

    # Verify device placement
    assert outputs['descriptors'].device.type == device.type
    assert outputs['scores'].device.type == device.type


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
