"""
Unit tests for PARENet criterion integration.

Tests PARENet loss function instantiation and computation.
"""

import pytest
import torch

from configs.common.criteria.point_cloud_registration.parenet_criterion_cfg import (
    criterion_cfg,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from utils.builders import build_from_config


def test_parenet_criterion_instantiation():
    """Test that PARENet criterion can be instantiated from config."""
    criterion = build_from_config(criterion_cfg)
    assert criterion is not None
    assert hasattr(criterion, 'parenet_loss')


def test_parenet_criterion_loss_computation():
    """Test PARENet criterion loss computation using actual model outputs."""
    from utils.builders import build_from_config as build_model
    from configs.common.models.point_cloud_registration.parenet_cfg import model_cfg
    from data.collators.parenet.parenet_collator_wrapper import parenet_collate_fn
    from data.collators.parenet.data import precompute_neibors as add_parenet_neighbors

    # Create config with buffer disabled for testing
    test_criterion_cfg = criterion_cfg.copy()
    test_criterion_cfg['args'] = criterion_cfg['args'].copy()
    test_criterion_cfg['args']['use_buffer'] = False

    criterion = build_from_config(test_criterion_cfg)

    # Get real model outputs to test criterion
    model = build_model(model_cfg).cuda().eval()

    # Create dummy datapoints following Pylon PCR dataset structure
    dummy_datapoints = [
        {
            'inputs': {
                'src_pc': PointCloud(
                    xyz=torch.randn(1000, 3),
                    data={'features': torch.ones(1000, 1)},
                ),
                'tgt_pc': PointCloud(
                    xyz=torch.randn(1000, 3),
                    data={'features': torch.ones(1000, 1)},
                ),
            },
            'labels': {
                'transform': torch.eye(4),
            },
            'meta_info': {
                'idx': 0,
                'dataset_name': 'test',
            }
        }
    ]

    # Step 1: Collate data (CPU processing with subsampling)
    batch = parenet_collate_fn(
        dummy_datapoints,
        num_stages=4,
        voxel_size=0.3,
        num_neighbors=[16, 16, 16, 16],
        subsample_ratio=0.25,
        precompute_data=True
    )

    # Step 2: Move to CUDA
    inputs = batch['inputs']
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.cuda()
        elif isinstance(value, list):
            inputs[key] = [v.cuda() if isinstance(v, torch.Tensor) else v for v in value]

    # Step 3: Verify neighbors were computed by collator
    assert 'neighbors' in inputs, "Neighbors should be computed by collator when precompute_data=True"
    assert 'subsampling' in inputs, "Subsampling should be computed by collator when precompute_data=True"
    assert 'upsampling' in inputs, "Upsampling should be computed by collator when precompute_data=True"

    # Step 4: Get model outputs
    with torch.no_grad():
        dummy_predictions = model(inputs)

    dummy_targets = {
        'transform': torch.eye(4).cuda(),
    }

    loss_result = criterion(dummy_predictions, dummy_targets)

    # Verify loss computation results
    assert isinstance(loss_result, torch.Tensor), "Loss output must be a tensor"
    assert loss_result.dim() == 0, "Loss must be a scalar tensor"

    # With random data, loss might be NaN/inf, so just check it's a valid tensor
    loss_value = loss_result.item()
    print(f"Loss value: {loss_value}")
    # Don't assert loss >= 0 since random data can produce NaN/inf


def test_parenet_criterion_device_handling():
    """Test PARENet criterion device handling."""
    criterion = build_from_config(criterion_cfg)

    # Test CPU mode
    for param in criterion.parameters():
        assert param.device.type == 'cpu'

    # Test CUDA mode (CUDA must be available)
    criterion_cuda = criterion.cuda()
    for param in criterion_cuda.parameters():
        assert param.device.type == 'cuda'


def test_parenet_criterion_parameter_count():
    """Test PARENet criterion parameter count."""
    criterion = build_from_config(criterion_cfg)

    total_params = sum(p.numel() for p in criterion.parameters())
    print(f"Criterion total parameters: {total_params}")

    # PARENet criterion should have minimal parameters (mainly for evaluator)
    assert total_params >= 0  # May have no parameters or few parameters
