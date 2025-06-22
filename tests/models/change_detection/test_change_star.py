"""Test ChangeStar model (your implementation)."""
import pytest
import torch
import torch.nn as nn
from typing import Dict
from models.change_detection.change_star.change_star import ChangeStar
from tests.models.utils.change_detection_data import generate_change_detection_data


@pytest.fixture
def encoder():
    """Create a simple encoder for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, 1, 1),  # Single image input channels
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, 2, 1),  # Downsample
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, 1, 1),  # Final feature maps
        nn.ReLU()
    )


@pytest.fixture
def change_decoder_cfg():
    """Configuration for change decoder."""
    return {
        'in_channels': 512,  # Concatenated features: 256 + 256
        'mid_channels': 128,
        'out_channels': 1,  # Binary change detection
        'drop_rate': 0.1,
        'scale_factor': 2.0,  # Upsample to match encoder downsampling
        'num_convs': 3
    }


@pytest.fixture
def semantic_decoder_cfg():
    """Configuration for semantic decoder."""
    return {
        'in_channels': 256,
        'out_channels': 10,  # Number of semantic classes
        'scale_factor': 2.0
    }


def get_change_star_model(encoder=None, change_decoder_cfg=None, semantic_decoder_cfg=None):
    """Get ChangeStar model instance."""
    if encoder is None:
        encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU()
        )

    if change_decoder_cfg is None:
        change_decoder_cfg = {
            'in_channels': 512,  # Concatenated features: 256 + 256
            'mid_channels': 128,
            'out_channels': 1,
            'drop_rate': 0.1,
            'scale_factor': 2.0,
            'num_convs': 3
        }

    if semantic_decoder_cfg is None:
        semantic_decoder_cfg = {
            'in_channels': 256,
            'out_channels': 10,
            'scale_factor': 2.0
        }

    return ChangeStar(
        encoder=encoder,
        change_decoder_cfg=change_decoder_cfg,
        semantic_decoder_cfg=semantic_decoder_cfg
    )


def test_change_star_model_initialization():
    """Test model initialization."""
    model = get_change_star_model()
    assert isinstance(model, nn.Module)
    assert hasattr(model, 'encoder')
    assert hasattr(model, 'change_decoder')
    assert hasattr(model, 'semantic_decoder')


def test_change_star_model_device_placement():
    """Test model device placement."""
    model = get_change_star_model()

    # Test CPU placement
    model = model.cpu()
    for param in model.parameters():
        assert param.device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_change_star_model_cuda_placement():
    """Test model CUDA placement."""
    model = get_change_star_model()

    # Test GPU placement
    model = model.cuda()
    for param in model.parameters():
        assert param.device.type == 'cuda'


def test_change_star_model_eval_mode():
    """Test model evaluation mode."""
    model = get_change_star_model()
    model.eval()

    assert not model.training
    for module in model.modules():
        if hasattr(module, 'training'):
            assert not module.training


def test_change_star_model_parameter_count():
    """Test model has reasonable parameter count."""
    model = get_change_star_model()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params <= total_params


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_change_star_cuda_device_consistency():
    """Test model and input device consistency."""
    model = get_change_star_model().cuda()
    dummy_input = generate_change_detection_data(batch_size=1, height=64, width=64)

    # Move input to GPU
    gpu_input = {k: v.cuda() for k, v in dummy_input.items()}

    model.eval()
    with torch.no_grad():
        output = model(gpu_input)

    # Check outputs are on GPU (model is in eval mode)
    assert output['change'].is_cuda
    assert output['semantic_1'].is_cuda
    assert output['semantic_2'].is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_change_star_gpu_memory_usage():
    """Test GPU memory usage is reasonable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()

    model = get_change_star_model().cuda()
    dummy_input = generate_change_detection_data(batch_size=2, height=128, width=128)
    gpu_input = {k: v.cuda() for k, v in dummy_input.items()}

    model.eval()
    with torch.no_grad():
        output = model(gpu_input)

    peak_memory = torch.cuda.memory_allocated()
    memory_used = peak_memory - initial_memory

    # Should use less than 2GB for this test
    assert memory_used < 2 * 1024**3


def test_change_star_memory_leak_detection():
    """Test for memory leaks in repeated forward passes."""
    model = get_change_star_model()
    dummy_input = generate_change_detection_data(batch_size=1, height=32, width=32)

    model.eval()

    # Run multiple forward passes
    for _ in range(10):
        with torch.no_grad():
            output = model(dummy_input)
        del output


def test_change_star_wrong_input_type():
    """Test model with wrong input type."""
    model = get_change_star_model()

    with pytest.raises((TypeError, AttributeError)):
        model("invalid_input")


def test_change_star_missing_input_keys():
    """Test model with missing input keys."""
    model = get_change_star_model()

    # Missing 'img_2' key
    incomplete_input = {'img_1': torch.randn(1, 3, 64, 64, dtype=torch.float32)}

    with pytest.raises(KeyError):
        model(incomplete_input)


def test_change_star_wrong_tensor_shape():
    """Test model with wrong tensor shapes."""
    model = get_change_star_model()

    # Wrong number of channels
    wrong_input = {
        'img_1': torch.randn(1, 1, 64, 64, dtype=torch.float32),  # Should be 3 channels
        'img_2': torch.randn(1, 1, 64, 64, dtype=torch.float32)
    }

    with pytest.raises((RuntimeError, ValueError)):
        model(wrong_input)


def test_change_star_gradient_flow_basic():
    """Test basic gradient flow through model."""
    model = get_change_star_model()
    model.train()  # Ensure training mode
    dummy_input = generate_change_detection_data(batch_size=1, height=64, width=64)

    # Enable gradients
    for tensor in dummy_input.values():
        tensor.requires_grad_(True)

    output = model(dummy_input)

    # Compute dummy loss and backpropagate
    loss = output['change_12'].sum() + output['change_21'].sum() + output['semantic'].sum()
    loss.backward()

    # Check gradients exist for model parameters
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    assert has_gradients


def test_change_star_gradient_accumulation():
    """Test gradient accumulation works correctly."""
    model = get_change_star_model()
    dummy_input = generate_change_detection_data(batch_size=1, height=32, width=32)

    # First forward pass
    output1 = model(dummy_input)
    loss1 = output1['change_12'].sum()
    loss1.backward()

    # Store gradients
    grad1 = {name: param.grad.clone() if param.grad is not None else None
             for name, param in model.named_parameters()}

    # Second forward pass (accumulate)
    output2 = model(dummy_input)
    loss2 = output2['change_12'].sum()
    loss2.backward()

    # Check gradients accumulated
    for name, param in model.named_parameters():
        if param.grad is not None and grad1[name] is not None:
            expected_grad = grad1[name] + grad1[name]  # Should be 2x original
            # Allow some numerical tolerance
            assert torch.allclose(param.grad, expected_grad, rtol=1e-4)


def test_change_star_forward_pass_basic():
    """Test basic forward pass functionality."""
    model = get_change_star_model()
    dummy_input = generate_change_detection_data(batch_size=2, height=64, width=64)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Check output structure (eval mode)
    assert isinstance(output, dict)
    assert 'change' in output
    assert 'semantic_1' in output
    assert 'semantic_2' in output

    # Check output shapes
    batch_size = 2
    assert output['change'].shape[0] == batch_size
    assert output['semantic_1'].shape[0] == batch_size
    assert output['semantic_2'].shape[0] == batch_size


def test_change_star_forward_pass_deterministic():
    """Test forward pass is deterministic in eval mode."""
    model = get_change_star_model()
    dummy_input = generate_change_detection_data(batch_size=1, height=64, width=64)

    model.eval()

    with torch.no_grad():
        output1 = model(dummy_input)
        output2 = model(dummy_input)

    # Outputs should be identical in eval mode
    assert torch.allclose(output1['change'], output2['change'])
    assert torch.allclose(output1['semantic_1'], output2['semantic_1'])
    assert torch.allclose(output1['semantic_2'], output2['semantic_2'])


def test_change_star_forward_pass_different_inputs():
    """Test forward pass with different inputs produces different outputs."""
    model = get_change_star_model()

    input1 = generate_change_detection_data(batch_size=1, height=64, width=64)
    input2 = generate_change_detection_data(batch_size=1, height=64, width=64)

    model.eval()

    with torch.no_grad():
        output1 = model(input1)
        output2 = model(input2)

    # Different inputs should produce different outputs
    assert not torch.allclose(output1['change'], output2['change'])
    assert not torch.allclose(output1['semantic_1'], output2['semantic_1'])


def test_change_star_initialization_with_fixtures(encoder, change_decoder_cfg, semantic_decoder_cfg):
    """Test ChangeStar initialization with fixtures."""
    model = get_change_star_model(
        encoder=encoder,
        change_decoder_cfg=change_decoder_cfg,
        semantic_decoder_cfg=semantic_decoder_cfg
    )

    assert model.encoder is encoder
    # Basic functionality test
    dummy_input = generate_change_detection_data(batch_size=1, height=64, width=64)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    assert 'change' in output
    assert 'semantic_1' in output
    assert 'semantic_2' in output


def test_change_star_multi_task_output_structure():
    """Test that ChangeStar produces multi-task outputs."""
    model = get_change_star_model()
    dummy_input = generate_change_detection_data(batch_size=2, height=64, width=64)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Should have all three outputs
    assert 'change' in output
    assert 'semantic_1' in output
    assert 'semantic_2' in output

    # Check that outputs have correct shapes
    batch_size = 2
    height, width = 64, 64

    # Change output should be binary (1 channel)
    assert output['change'].shape == (batch_size, 1, height, width)

    # Semantic outputs should have multiple classes
    assert output['semantic_1'].shape[0] == batch_size
    assert output['semantic_1'].shape[1] > 1  # Multiple classes
    assert output['semantic_1'].shape[2:] == (height, width)
    assert output['semantic_2'].shape[0] == batch_size
    assert output['semantic_2'].shape[1] > 1  # Multiple classes
    assert output['semantic_2'].shape[2:] == (height, width)


@pytest.mark.parametrize("input_size", [(32, 32), (64, 64), (128, 128)])
def test_change_star_different_input_sizes(input_size):
    """Test model with different input sizes."""
    model = get_change_star_model()
    height, width = input_size

    dummy_input = generate_change_detection_data(batch_size=1, height=height, width=width)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Output should match input spatial dimensions (eval mode)
    assert output['change'].shape[2:] == (height, width)
    assert output['semantic_1'].shape[2:] == (height, width)
    assert output['semantic_2'].shape[2:] == (height, width)


@pytest.mark.parametrize("drop_rate", [0.0, 0.1, 0.3, 0.5])
def test_change_star_drop_connect_functionality(drop_rate):
    """Test model with different drop rates."""
    change_decoder_cfg = {
        'in_channels': 512,  # Concatenated features: 256 + 256
        'mid_channels': 128,
        'out_channels': 1,
        'drop_rate': drop_rate,
        'scale_factor': 2.0,
        'num_convs': 3
    }

    model = get_change_star_model(change_decoder_cfg=change_decoder_cfg)
    dummy_input = generate_change_detection_data(batch_size=1, height=64, width=64)

    # Test in training mode (dropout active)
    model.train()
    output_train = model(dummy_input)

    # Test in eval mode (dropout inactive)
    model.eval()
    with torch.no_grad():
        output_eval = model(dummy_input)

    # Basic checks (compare training vs eval mode outputs)
    assert output_train['change_12'].shape == output_eval['change'].shape
    assert output_train['semantic'].shape == output_eval['semantic_1'].shape

    # For non-zero dropout, outputs should be different between train/eval
    if drop_rate > 0:
        assert not torch.allclose(output_train['change_12'], output_eval['change'], atol=1e-6)


def test_change_star_gradient_flow_multi_task():
    """Test gradient flow through both tasks."""
    model = get_change_star_model()
    model.train()  # Ensure training mode
    dummy_input = generate_change_detection_data(batch_size=1, height=64, width=64)

    output = model(dummy_input)

    # Compute loss for both tasks
    change_loss = output['change_12'].sum() + output['change_21'].sum()
    semantic_loss = output['semantic'].sum()
    total_loss = change_loss + semantic_loss

    total_loss.backward()

    # Check that gradients flow to all parts of the model
    encoder_has_grad = any(p.grad is not None for p in model.encoder.parameters())
    change_decoder_has_grad = any(p.grad is not None for p in model.change_decoder.parameters())
    semantic_decoder_has_grad = any(p.grad is not None for p in model.semantic_decoder.parameters())

    assert encoder_has_grad
    assert change_decoder_has_grad
    assert semantic_decoder_has_grad


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_change_star_different_batch_sizes(batch_size):
    """Test model with different batch sizes."""
    model = get_change_star_model()
    dummy_input = generate_change_detection_data(batch_size=batch_size, height=64, width=64)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Check batch dimension in outputs (eval mode)
    assert output['change'].shape[0] == batch_size
    assert output['semantic_1'].shape[0] == batch_size
    assert output['semantic_2'].shape[0] == batch_size


def test_change_star_model_state_dict():
    """Test model state dict functionality."""
    model = get_change_star_model()

    # Get state dict
    state_dict = model.state_dict()
    assert isinstance(state_dict, dict)
    assert len(state_dict) > 0

    # Create new model and load state dict
    new_model = get_change_star_model()
    new_model.load_state_dict(state_dict)

    # Test that loaded model works
    dummy_input = generate_change_detection_data(batch_size=1, height=64, width=64)

    new_model.eval()
    with torch.no_grad():
        output = new_model(dummy_input)

    assert 'change' in output
    assert 'semantic_1' in output
    assert 'semantic_2' in output
