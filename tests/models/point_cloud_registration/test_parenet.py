"""
Unit tests for PARENet model integration.

Tests PARENet model instantiation, forward pass, and integration with Pylon framework.
"""

import pytest
import torch
from utils.builders import build_from_config
from configs.common.models.point_cloud_registration.parenet_cfg import model_cfg


def test_parenet_model_instantiation():
    """Test that PARENet model can be instantiated from config."""
    model = build_from_config(model_cfg)
    assert model is not None
    assert hasattr(model, 'parenet_model')
    assert hasattr(model, 'cfg')


def test_parenet_model_forward_pass():
    """Test PARENet model forward pass with proper data pipeline."""
    from data.collators.parenet.parenet_collator_wrapper import parenet_collate_fn
    from data.collators.parenet.data import precompute_neibors as add_parenet_neighbors
    
    model = build_from_config(model_cfg)
    model = model.cuda()
    model.eval()
    
    # Create dummy datapoints following Pylon PCR dataset structure
    dummy_datapoints = [
        {
            'inputs': {
                'src_pc': {
                    'pos': torch.randn(1000, 3),
                    'features': torch.ones(1000, 1),
                },
                'tgt_pc': {
                    'pos': torch.randn(1000, 3),
                    'features': torch.ones(1000, 1),
                },
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
    
    # Step 4: Test forward pass
    with torch.no_grad():
        outputs = model(inputs)
    
    # Verify outputs
    assert isinstance(outputs, dict), "Model output must be dictionary"
    assert 'estimated_transform' in outputs, "Output must contain estimated_transform"
    
    estimated_transform = outputs['estimated_transform']
    assert isinstance(estimated_transform, torch.Tensor), "estimated_transform must be tensor"
    assert estimated_transform.shape == (4, 4), f"Expected shape (4, 4), got {estimated_transform.shape}"
    assert estimated_transform.device.type == 'cuda', "Output must be on CUDA"
    
    print(f"✓ PARENet forward pass successful, output shape: {estimated_transform.shape}")


def test_parenet_model_device_handling():
    """Test PARENet model device handling."""
    model = build_from_config(model_cfg)
    
    # Test CPU mode
    assert next(model.parameters()).device.type == 'cpu'
    
    # Test CUDA mode (CUDA must be available)
    model_cuda = model.cuda()
    assert next(model_cuda.parameters()).device.type == 'cuda'


def test_parenet_model_parameter_count():
    """Test PARENet model has reasonable parameter count."""
    model = build_from_config(model_cfg)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # PARENet should have a significant number of parameters (> 1M)
    assert total_params > 1_000_000, f"Model has too few parameters: {total_params}"
    assert trainable_params > 1_000_000, f"Model has too few trainable parameters: {trainable_params}"
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


def test_parenet_model_backbone_structure():
    """Test PARENet model backbone structure."""
    model = build_from_config(model_cfg)
    
    # Check that the model has the expected backbone structure
    assert hasattr(model.parenet_model, 'backbone')
    backbone = model.parenet_model.backbone
    
    # Check backbone has expected components
    expected_backbone_attrs = ['encoder2_1', 'encoder2_2', 'encoder2_3']
    for attr in expected_backbone_attrs:
        assert hasattr(backbone, attr), f"Backbone missing {attr}"
