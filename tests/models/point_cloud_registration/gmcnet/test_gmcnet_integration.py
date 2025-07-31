"""Integration tests for GMCNet wrapper functionality.

This module tests the integration between the GMCNet wrapper and the Pylon framework,
focusing specifically on GMCNet functionality. Tests will fail immediately if GMCNet
dependencies are not properly installed - no fallback logic or defensive programming.
"""

from typing import Dict
import pytest
import torch

from models.point_cloud_registration.gmcnet.gmcnet_wrapper import GMCNet
from data.collators.base_collator import BaseCollator


def _create_sample_point_cloud_data(batch_size: int = 2, num_points: int = 512) -> Dict[str, torch.Tensor]:
    """Create sample point cloud data for GMCNet testing."""
    # Generate base point cloud (normalized to [-1, 1] for stable testing)
    src_points = 2 * torch.rand(batch_size, num_points, 3, dtype=torch.float32) - 1
    
    # Create realistic transformation matrix
    transform = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Add small rotation around z-axis
    angle = 0.1  # Small rotation angle for stable testing
    cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    rotation = torch.tensor([
        [cos_a, -sin_a, 0, 0],
        [sin_a, cos_a, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)
    transform[0] = rotation
    
    # Add small translation
    transform[:, :3, 3] = torch.randn(batch_size, 3) * 0.1
    
    # Apply transformation to create target points
    homogeneous_src = torch.cat([src_points, torch.ones(batch_size, num_points, 1)], dim=2)
    transformed_points = torch.bmm(transform[:, :3, :], homogeneous_src.transpose(1, 2)).transpose(1, 2)
    tgt_points = transformed_points + torch.randn_like(transformed_points) * 0.01  # Add small noise
    
    return {
        'src_points': src_points,
        'tgt_points': tgt_points,
        'transform': transform
    }


@pytest.fixture
def gmcnet_args():
    """Create GMCNet arguments for testing."""
    class TestArgs:
        def __init__(self):
            # Model architecture parameters
            self.emb_dims = 256
            self.n_blocks = 4
            self.n_heads = 4
            self.ff_dims = 256
            self.dropout = 0.0
            
            # Point cloud parameters
            self.n_points = 512
            self.n_subsampled_points = 384
            
            # Loss parameters
            self.loss_type = 'mae'
            
            # Training parameters  
            self.noise_type = 'crop'
            self.partial_p_keep = [0.7, 0.7]
            
            # Disable computationally expensive features for testing
            self.use_rri = False
            
    return TestArgs()


@pytest.fixture
def gmcnet_model(gmcnet_args):
    """Create GMCNet model for testing.
    
    This fixture creates a real GMCNet model. If dependencies are missing
    or the model cannot be instantiated, the test will fail immediately.
    No fallback logic - fail fast and loud.
    """
    # Create real GMCNet model - will fail if dependencies missing
    model = GMCNet(gmcnet_args)
    
    # Validate model was created properly
    assert hasattr(model, '_model'), "GMCNet wrapper must have _model attribute"
    assert isinstance(model, torch.nn.Module), "GMCNet wrapper must be a PyTorch module"
    assert hasattr(model, 'forward'), "GMCNet wrapper must have forward method"
    assert callable(model.forward), "GMCNet wrapper forward method must be callable"
    
    return model


def test_gmcnet_wrapper_initialization(gmcnet_model):
    """Test GMCNet wrapper can be initialized with proper arguments."""
    model = gmcnet_model
    
    # Verify initialization - these assertions will fail if model is not properly initialized
    assert model is not None, "GMCNet model must not be None"
    assert hasattr(model, 'args'), "GMCNet wrapper must have args attribute"
    assert hasattr(model, '_model'), "GMCNet wrapper must have _model attribute"
    assert isinstance(model, torch.nn.Module), "GMCNet wrapper must inherit from torch.nn.Module"


def test_gmcnet_wrapper_train_mode_forward(gmcnet_model):
    """Test GMCNet wrapper forward pass in training mode with ground truth."""
    model = gmcnet_model
    batch_size = 2
    num_points = 512
    
    # Create sample data
    sample_data = _create_sample_point_cloud_data(batch_size, num_points)
    inputs = {
        'src_points': sample_data['src_points'],
        'tgt_points': sample_data['tgt_points'],
        'transform': sample_data['transform'],
        'mode': 'train'
    }
    
    # Forward pass - will fail if GMCNet implementation has issues
    outputs = model(inputs)
    
    # Validate training outputs - strict assertions, no defensive checks
    assert 'loss' in outputs, "Training mode must produce loss output"
    assert 'transformation' in outputs, "Training mode must produce transformation output"
    assert 'T_12' in outputs, "Training mode must produce T_12 output"
    
    # Check output types and shapes - fail fast if wrong
    assert isinstance(outputs['loss'], torch.Tensor), "Loss must be a tensor"
    assert outputs['loss'].numel() == 1, "Loss must be scalar"
    assert outputs['transformation'].shape == (batch_size, 4, 4), f"Transformation shape must be ({batch_size}, 4, 4)"
    assert torch.equal(outputs['T_12'], outputs['transformation']), "T_12 and transformation must be identical"
    
    # Check that gradient is enabled for loss
    assert outputs['loss'].requires_grad, "Loss tensor must require gradients for training"


def test_gmcnet_wrapper_test_mode_forward(gmcnet_model):
    """Test GMCNet wrapper forward pass in test mode without ground truth."""
    model = gmcnet_model
    batch_size = 2
    num_points = 512
    
    # Create sample data (no ground truth transform needed)
    sample_data = _create_sample_point_cloud_data(batch_size, num_points)
    inputs = {
        'src_points': sample_data['src_points'],
        'tgt_points': sample_data['tgt_points'],
        'mode': 'test'
    }
    
    # Forward pass
    outputs = model(inputs)
    
    # Validate test outputs - strict assertions
    assert 'transformation' in outputs, "Test mode must produce transformation output"
    assert 'T_12' in outputs, "Test mode must produce T_12 output"
    
    # Check output shapes
    assert outputs['transformation'].shape == (batch_size, 4, 4), f"Transformation shape must be ({batch_size}, 4, 4)"
    assert torch.equal(outputs['T_12'], outputs['transformation']), "T_12 and transformation must be identical"
    
    # Test mode should not include loss
    assert 'loss' not in outputs, "Test mode must not produce loss output"


def test_gmcnet_wrapper_input_validation(gmcnet_model):
    """Test GMCNet wrapper input validation and error handling."""
    model = gmcnet_model
    sample_data = _create_sample_point_cloud_data(2, 512)
    
    # Test missing src_points - should fail fast
    inputs_missing_src = {
        'tgt_points': sample_data['tgt_points'],
        'mode': 'test'
    }
    
    with pytest.raises((AssertionError, KeyError, ValueError), match="src_points"):
        model(inputs_missing_src)
    
    # Test missing tgt_points - should fail fast
    inputs_missing_tgt = {
        'src_points': sample_data['src_points'],
        'mode': 'test'
    }
    
    with pytest.raises((AssertionError, KeyError, ValueError), match="tgt_points"):
        model(inputs_missing_tgt)
    
    # Test missing transform in train mode - should fail fast
    inputs_train_no_transform = {
        'src_points': sample_data['src_points'],
        'tgt_points': sample_data['tgt_points'],
        'mode': 'train'
    }
    
    with pytest.raises((ValueError, KeyError, AssertionError)):
        model(inputs_train_no_transform)


def test_gmcnet_wrapper_api_methods(gmcnet_model):
    """Test GMCNet wrapper API methods (get_transform, visualize)."""
    model = gmcnet_model
    batch_size = 2
    num_points = 512
    
    # Run forward pass first to populate internal state
    sample_data = _create_sample_point_cloud_data(batch_size, num_points)
    inputs = {
        'src_points': sample_data['src_points'],
        'tgt_points': sample_data['tgt_points'],
        'transform': sample_data['transform'],
        'mode': 'train'
    }
    
    model(inputs)
    
    # Test get_transform method - should work or fail clearly
    transform_result = model.get_transform()
    assert transform_result is not None, "get_transform must return a result"
    assert isinstance(transform_result, torch.Tensor), "get_transform must return a tensor"
    assert transform_result.shape == (batch_size, 4, 4), f"Transform shape must be ({batch_size}, 4, 4)"
    
    # Test visualize method - should work or fail clearly
    viz_result = model.visualize(0)
    assert viz_result is not None, "visualize must return a result"


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_gmcnet_wrapper_batch_size_handling(gmcnet_model, batch_size):
    """Test GMCNet wrapper handles different batch sizes correctly."""
    model = gmcnet_model
    num_points = 256
    
    # Create sample data with specified batch size
    sample_data = _create_sample_point_cloud_data(batch_size, num_points)
    
    inputs = {
        'src_points': sample_data['src_points'],
        'tgt_points': sample_data['tgt_points'],
        'transform': sample_data['transform'],
        'mode': 'train'
    }
    
    # Test model forward pass - should handle all batch sizes or fail clearly
    outputs = model(inputs)
    assert 'transformation' in outputs, "Model must produce transformation output"
    assert outputs['transformation'].shape[0] == batch_size, f"Batch dimension must be {batch_size}"
    assert outputs['transformation'].shape == (batch_size, 4, 4), f"Output shape must be ({batch_size}, 4, 4)"


@pytest.mark.parametrize("num_points", [256, 512, 1024])
def test_gmcnet_wrapper_point_count_handling(gmcnet_model, num_points):
    """Test GMCNet wrapper handles different point cloud sizes."""
    model = gmcnet_model
    batch_size = 2
    
    # Create sample data with specified point count
    sample_data = _create_sample_point_cloud_data(batch_size, num_points)
    
    inputs = {
        'src_points': sample_data['src_points'],
        'tgt_points': sample_data['tgt_points'],
        'transform': sample_data['transform'],
        'mode': 'train'
    }
    
    # Verify input shapes are correct
    assert inputs['src_points'].shape == (batch_size, num_points, 3), f"Source points shape must be ({batch_size}, {num_points}, 3)"
    assert inputs['tgt_points'].shape == (batch_size, num_points, 3), f"Target points shape must be ({batch_size}, {num_points}, 3)"
    
    # Test forward pass - should handle all point counts or fail clearly
    outputs = model(inputs)
    
    # Verify output format is consistent regardless of input point count
    assert 'transformation' in outputs, "Model must produce transformation output"
    assert outputs['transformation'].shape == (batch_size, 4, 4), f"Transformation shape must be ({batch_size}, 4, 4)"


def test_gmcnet_wrapper_with_base_collator(gmcnet_model):
    """Test GMCNet wrapper integration with Pylon's BaseCollator."""
    model = gmcnet_model
    
    # Create batch data that BaseCollator would produce
    batch_data = []
    for i in range(2):
        sample_data = _create_sample_point_cloud_data(1, 256)
        datapoint = (
            {  # inputs
                'src_points': sample_data['src_points'].squeeze(0),
                'tgt_points': sample_data['tgt_points'].squeeze(0)
            },
            {  # labels
                'transform': sample_data['transform'].squeeze(0)
            },
            {  # meta_info
                'index': i,
                'sample_id': f'test_{i}'
            }
        )
        batch_data.append(datapoint)
    
    # Use BaseCollator to create batch
    collator = BaseCollator()
    batch = collator(batch_data)
    
    # Extract data for GMCNet wrapper
    inputs = {
        'src_points': batch[0]['src_points'],  # inputs dict
        'tgt_points': batch[0]['tgt_points'],
        'transform': batch[1]['transform'],    # labels dict  
        'mode': 'train'
    }
    
    # Test that wrapper works with collated data - no fallback handling
    outputs = model(inputs)
    
    # Verify outputs - strict assertions
    assert 'transformation' in outputs, "Model must produce transformation output"
    assert 'loss' in outputs, "Training mode must produce loss output"
    assert outputs['transformation'].shape == (2, 4, 4), "Transformation shape must match batch size (2, 4, 4)"


def test_gmcnet_wrapper_pytorch_module_interface(gmcnet_model):
    """Test that GMCNet wrapper properly implements PyTorch module interface."""
    model = gmcnet_model
    
    # Test that wrapper is a proper PyTorch module - strict checks
    assert isinstance(model, torch.nn.Module), "GMCNet wrapper must inherit from torch.nn.Module"
    assert hasattr(model, 'forward'), "GMCNet wrapper must have forward method"
    assert callable(model.forward), "GMCNet wrapper forward method must be callable"
    assert hasattr(model, 'parameters'), "GMCNet wrapper must have parameters method"
    assert hasattr(model, 'named_parameters'), "GMCNet wrapper must have named_parameters method"
    
    # Test that parameters exist for real model
    params = list(model.parameters())
    named_params = list(model.named_parameters())
    
    # Real GMCNet should have parameters - no defensive checks
    assert len(params) > 0, "GMCNet must have trainable parameters"
    assert len(named_params) > 0, "GMCNet must have named parameters"
    
    # Test training/eval mode switching
    model.train()
    assert model.training, "Model must be in training mode after calling train()"
    
    model.eval()
    assert not model.training, "Model must be in eval mode after calling eval()"
    
    # Test device transfer (if CUDA available)
    if torch.cuda.is_available():
        cuda_model = model.cuda()
        assert next(cuda_model.parameters()).is_cuda, "Model parameters must be on CUDA after cuda() call"
        
        cpu_model = cuda_model.cpu()
        assert not next(cpu_model.parameters()).is_cuda, "Model parameters must be on CPU after cpu() call"


def test_gmcnet_wrapper_output_format_consistency(gmcnet_model):
    """Test that GMCNet wrapper produces consistent output formats across modes."""
    model = gmcnet_model
    
    # Test output consistency across different batch sizes
    for batch_size in [1, 2]:
        sample_data = _create_sample_point_cloud_data(batch_size, 256)
        
        # Test training mode
        train_inputs = {
            'src_points': sample_data['src_points'],
            'tgt_points': sample_data['tgt_points'],
            'transform': sample_data['transform'],
            'mode': 'train'
        }
        
        train_outputs = model(train_inputs)
        assert 'transformation' in train_outputs, "Training mode must produce transformation"
        assert 'T_12' in train_outputs, "Training mode must produce T_12"
        assert 'loss' in train_outputs, "Training mode must produce loss"
        assert train_outputs['transformation'].shape == (batch_size, 4, 4), f"Training transformation shape must be ({batch_size}, 4, 4)"
        assert torch.equal(train_outputs['T_12'], train_outputs['transformation']), "T_12 and transformation must be identical in training mode"
        
        # Test test mode
        test_inputs = {
            'src_points': sample_data['src_points'],
            'tgt_points': sample_data['tgt_points'],
            'mode': 'test'
        }
        
        test_outputs = model(test_inputs)
        assert 'transformation' in test_outputs, "Test mode must produce transformation"
        assert 'T_12' in test_outputs, "Test mode must produce T_12"
        assert test_outputs['transformation'].shape == (batch_size, 4, 4), f"Test transformation shape must be ({batch_size}, 4, 4)"
        assert torch.equal(test_outputs['T_12'], test_outputs['transformation']), "T_12 and transformation must be identical in test mode"
        
        # Test mode should not have loss
        assert 'loss' not in test_outputs, "Test mode must not produce loss"


def test_gmcnet_wrapper_tensor_dtypes_and_devices(gmcnet_model):
    """Test GMCNet wrapper handles tensor dtypes and devices correctly."""
    model = gmcnet_model
    batch_size = 2
    num_points = 256
    
    # Test with float32 tensors (standard)
    sample_data = _create_sample_point_cloud_data(batch_size, num_points)
    
    inputs = {
        'src_points': sample_data['src_points'],
        'tgt_points': sample_data['tgt_points'],
        'transform': sample_data['transform'],
        'mode': 'train'
    }
    
    # Verify input dtypes
    assert inputs['src_points'].dtype == torch.float32, "Source points must be float32"
    assert inputs['tgt_points'].dtype == torch.float32, "Target points must be float32"
    assert inputs['transform'].dtype == torch.float32, "Transform must be float32"
    
    # Forward pass
    outputs = model(inputs)
    
    # Verify output dtypes match inputs
    assert outputs['transformation'].dtype == torch.float32, "Output transformation must be float32"
    assert outputs['loss'].dtype == torch.float32, "Output loss must be float32"
    
    # Test device consistency
    device = inputs['src_points'].device
    assert outputs['transformation'].device == device, "Output transformation must be on same device as inputs"
    assert outputs['loss'].device == device, "Output loss must be on same device as inputs"


def test_gmcnet_wrapper_gradient_flow(gmcnet_model):
    """Test that GMCNet wrapper maintains proper gradient flow."""
    model = gmcnet_model
    model.train()  # Ensure model is in training mode
    
    batch_size = 2
    num_points = 256
    
    sample_data = _create_sample_point_cloud_data(batch_size, num_points)
    
    inputs = {
        'src_points': sample_data['src_points'],
        'tgt_points': sample_data['tgt_points'],
        'transform': sample_data['transform'],
        'mode': 'train'
    }
    
    # Forward pass
    outputs = model(inputs)
    
    # Verify loss requires gradients
    assert outputs['loss'].requires_grad, "Loss must require gradients for backpropagation"
    
    # Test backward pass
    loss = outputs['loss']
    loss.backward()
    
    # Verify gradients were computed for model parameters
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    assert has_gradients, "At least some model parameters must have non-zero gradients after backward pass"
