"""Test CSA_CDGAN model (your implementation)."""
import pytest
import torch
import torch.nn as nn
from typing import Dict
from models.change_detection.csa_cdgan.model import CSA_CDGAN
from tests.models.utils.change_detection_data import generate_change_detection_data


class SimpleGenerator(nn.Module):
    """Simple generator for testing."""

    def __init__(self, output_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 64, 3, 1, 1)  # Concatenated input
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, output_channels, 3, 1, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        img1 = inputs['img_1']
        img2 = inputs['img_2']
        x = torch.cat([img1, img2], dim=1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x


class SimpleDiscriminator(nn.Module):
    """Simple discriminator for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(7, 64, 3, 1, 1)  # 6 + 1 for change map
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 16, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Dict[str, torch.Tensor], change_map: torch.Tensor) -> torch.Tensor:
        img1 = inputs['img_1']
        img2 = inputs['img_2']
        x = torch.cat([img1, img2, change_map], dim=1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc(x))
        return x


@pytest.fixture
def generator_cfg():
    """Configuration for generator."""
    return {
        'class': SimpleGenerator,
        'args': {'output_channels': 1}
    }


@pytest.fixture
def discriminator_cfg():
    """Configuration for discriminator."""
    return {
        'class': SimpleDiscriminator,
        'args': {}
    }


def get_csa_cdgan_model(generator_cfg=None, discriminator_cfg=None):
    """Get CSA_CDGAN model instance."""
    if generator_cfg is None:
        generator_cfg = {
            'class': SimpleGenerator,
            'args': {'output_channels': 1}
        }

    if discriminator_cfg is None:
        discriminator_cfg = {
            'class': SimpleDiscriminator,
            'args': {}
        }

    return CSA_CDGAN(
        generator_cfg=generator_cfg,
        discriminator_cfg=discriminator_cfg
    )


def test_csa_cdgan_model_initialization():
    """Test model initialization."""
    model = get_csa_cdgan_model()
    assert isinstance(model, nn.Module)
    assert hasattr(model, 'generator')
    assert hasattr(model, 'discriminator')


def test_csa_cdgan_model_device_placement():
    """Test model device placement."""
    model = get_csa_cdgan_model()

    # Test CPU placement
    model = model.cpu()
    for param in model.parameters():
        assert param.device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_csa_cdgan_model_cuda_placement():
    """Test model CUDA placement."""
    model = get_csa_cdgan_model()

    # Test GPU placement
    model = model.cuda()
    for param in model.parameters():
        assert param.device.type == 'cuda'


def test_csa_cdgan_model_eval_mode():
    """Test model evaluation mode."""
    model = get_csa_cdgan_model()
    model.eval()

    assert not model.training
    assert not model.generator.training
    assert not model.discriminator.training


def test_csa_cdgan_model_parameter_count():
    """Test model has reasonable parameter count."""
    model = get_csa_cdgan_model()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params <= total_params


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_csa_cdgan_cuda_device_consistency():
    """Test model and input device consistency."""
    model = get_csa_cdgan_model().cuda()
    dummy_input = generate_change_detection_data(batch_size=1, height=64, width=64)

    # Move input to GPU
    gpu_input = {k: v.cuda() for k, v in dummy_input.items()}

    model.eval()
    with torch.no_grad():
        generator_output = model.generator(gpu_input)
        discriminator_output = model.discriminator(gpu_input, generator_output)

    # Check outputs are on GPU
    assert generator_output.is_cuda
    assert discriminator_output.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_csa_cdgan_gpu_memory_usage():
    """Test GPU memory usage is reasonable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()

    model = get_csa_cdgan_model().cuda()
    dummy_input = generate_change_detection_data(batch_size=2, height=128, width=128)
    gpu_input = {k: v.cuda() for k, v in dummy_input.items()}

    model.eval()
    with torch.no_grad():
        generator_output = model.generator(gpu_input)
        discriminator_output = model.discriminator(gpu_input, generator_output)

    peak_memory = torch.cuda.memory_allocated()
    memory_used = peak_memory - initial_memory

    # Should use less than 2GB for this test
    assert memory_used < 2 * 1024**3


def test_csa_cdgan_memory_leak_detection():
    """Test for memory leaks in repeated forward passes."""
    model = get_csa_cdgan_model()
    dummy_input = generate_change_detection_data(batch_size=1, height=32, width=32)

    model.eval()

    # Run multiple forward passes
    for _ in range(10):
        with torch.no_grad():
            generator_output = model.generator(dummy_input)
            discriminator_output = model.discriminator(dummy_input, generator_output)
        del generator_output, discriminator_output


def test_csa_cdgan_wrong_input_type():
    """Test model with wrong input type."""
    model = get_csa_cdgan_model()

    with pytest.raises((TypeError, AttributeError)):
        model.generator("invalid_input")


def test_csa_cdgan_missing_input_keys():
    """Test model with missing input keys."""
    model = get_csa_cdgan_model()

    # Missing 'img_2' key
    incomplete_input = {'img_1': torch.randn(1, 3, 64, 64, dtype=torch.float32)}

    with pytest.raises(KeyError):
        model.generator(incomplete_input)


def test_csa_cdgan_wrong_tensor_shape():
    """Test model with wrong tensor shapes."""
    model = get_csa_cdgan_model()

    # Wrong number of channels
    wrong_input = {
        'img_1': torch.randn(1, 1, 64, 64, dtype=torch.float32),  # Should be 3 channels
        'img_2': torch.randn(1, 1, 64, 64, dtype=torch.float32)
    }

    with pytest.raises((RuntimeError, ValueError)):
        model.generator(wrong_input)


def test_csa_cdgan_generator_forward_pass():
    """Test generator forward pass functionality."""
    model = get_csa_cdgan_model()
    dummy_input = generate_change_detection_data(batch_size=2, height=64, width=64)

    model.eval()
    with torch.no_grad():
        output = model.generator(dummy_input)

    # Check output shape
    batch_size = 2
    height, width = 64, 64
    assert output.shape == (batch_size, 1, height, width)  # Single channel change map

    # Check output range (should be in [0, 1] due to sigmoid)
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_csa_cdgan_discriminator_forward_pass():
    """Test discriminator forward pass functionality."""
    model = get_csa_cdgan_model()
    dummy_input = generate_change_detection_data(batch_size=2, height=64, width=64)
    dummy_change_map = torch.randn(2, 1, 64, 64, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model.discriminator(dummy_input, dummy_change_map)

    # Check output shape (should be batch_size x 1 for binary classification)
    batch_size = 2
    assert output.shape == (batch_size, 1)

    # Check output range (should be in [0, 1] due to sigmoid)
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_csa_cdgan_end_to_end_forward_pass():
    """Test end-to-end forward pass (generator -> discriminator)."""
    model = get_csa_cdgan_model()
    dummy_input = generate_change_detection_data(batch_size=2, height=64, width=64)

    model.eval()
    with torch.no_grad():
        # Generator produces change map
        generated_change_map = model.generator(dummy_input)

        # Discriminator evaluates the generated change map
        discriminator_output = model.discriminator(dummy_input, generated_change_map)

    # Check shapes
    batch_size = 2
    assert generated_change_map.shape == (batch_size, 1, 64, 64)
    assert discriminator_output.shape == (batch_size, 1)


def test_csa_cdgan_gradient_flow_generator():
    """Test gradient flow through generator."""
    model = get_csa_cdgan_model()
    dummy_input = generate_change_detection_data(batch_size=1, height=64, width=64)

    # Enable gradients for inputs
    for tensor in dummy_input.values():
        tensor.requires_grad_(True)

    output = model.generator(dummy_input)

    # Compute dummy loss and backpropagate
    loss = output.sum()
    loss.backward()

    # Check gradients exist for generator parameters
    has_gradients = False
    for param in model.generator.parameters():
        if param.grad is not None:
            has_gradients = True
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    assert has_gradients


def test_csa_cdgan_gradient_flow_discriminator():
    """Test gradient flow through discriminator."""
    model = get_csa_cdgan_model()
    dummy_input = generate_change_detection_data(batch_size=1, height=64, width=64)
    dummy_change_map = torch.randn(1, 1, 64, 64, dtype=torch.float32, requires_grad=True)

    # Enable gradients for inputs
    for tensor in dummy_input.values():
        tensor.requires_grad_(True)

    output = model.discriminator(dummy_input, dummy_change_map)

    # Compute dummy loss and backpropagate
    loss = output.sum()
    loss.backward()

    # Check gradients exist for discriminator parameters
    has_gradients = False
    for param in model.discriminator.parameters():
        if param.grad is not None:
            has_gradients = True
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    assert has_gradients


def test_csa_cdgan_adversarial_training_simulation():
    """Test simulated adversarial training (generator vs discriminator)."""
    model = get_csa_cdgan_model()
    dummy_input = generate_change_detection_data(batch_size=1, height=32, width=32)

    # Simulate generator training step
    generated_change_map = model.generator(dummy_input)
    discriminator_output = model.discriminator(dummy_input, generated_change_map)

    # Generator tries to fool discriminator (wants high discriminator output)
    generator_loss = -discriminator_output.mean()  # Adversarial loss
    generator_loss.backward(retain_graph=True)

    # Check generator has gradients
    generator_has_grad = any(p.grad is not None for p in model.generator.parameters())
    assert generator_has_grad

    # Clear generator gradients for discriminator training
    model.generator.zero_grad()

    # Simulate discriminator training step with real data
    real_change_map = torch.ones_like(generated_change_map)  # Fake "real" data
    real_output = model.discriminator(dummy_input, real_change_map)
    fake_output = model.discriminator(dummy_input, generated_change_map.detach())

    # Discriminator tries to distinguish real from fake
    discriminator_loss = -(real_output.mean() - fake_output.mean())  # Wasserstein-like loss
    discriminator_loss.backward()

    # Check discriminator has gradients
    discriminator_has_grad = any(p.grad is not None for p in model.discriminator.parameters())
    assert discriminator_has_grad


def test_csa_cdgan_initialization_with_fixtures(generator_cfg, discriminator_cfg):
    """Test CSA_CDGAN initialization with fixtures."""
    model = get_csa_cdgan_model(
        generator_cfg=generator_cfg,
        discriminator_cfg=discriminator_cfg
    )

    # Basic functionality test
    dummy_input = generate_change_detection_data(batch_size=1, height=64, width=64)

    model.eval()
    with torch.no_grad():
        generator_output = model.generator(dummy_input)
        discriminator_output = model.discriminator(dummy_input, generator_output)

    assert generator_output.shape == (1, 1, 64, 64)
    assert discriminator_output.shape == (1, 1)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_csa_cdgan_different_batch_sizes(batch_size):
    """Test model with different batch sizes."""
    model = get_csa_cdgan_model()
    dummy_input = generate_change_detection_data(batch_size=batch_size, height=64, width=64)

    model.eval()
    with torch.no_grad():
        generator_output = model.generator(dummy_input)
        discriminator_output = model.discriminator(dummy_input, generator_output)

    # Check batch dimension in outputs
    assert generator_output.shape[0] == batch_size
    assert discriminator_output.shape[0] == batch_size


@pytest.mark.parametrize("input_size", [(32, 32), (64, 64), (128, 128)])
def test_csa_cdgan_different_input_sizes(input_size):
    """Test model with different input sizes."""
    model = get_csa_cdgan_model()
    height, width = input_size

    dummy_input = generate_change_detection_data(batch_size=1, height=height, width=width)

    model.eval()
    with torch.no_grad():
        generator_output = model.generator(dummy_input)
        discriminator_output = model.discriminator(dummy_input, generator_output)

    # Generator output should match input spatial dimensions
    assert generator_output.shape[2:] == (height, width)
    # Discriminator output should be scalar per batch
    assert discriminator_output.shape == (1, 1)


def test_csa_cdgan_model_state_dict():
    """Test model state dict functionality."""
    model = get_csa_cdgan_model()

    # Get state dict
    state_dict = model.state_dict()
    assert isinstance(state_dict, dict)
    assert len(state_dict) > 0

    # Create new model and load state dict
    new_model = get_csa_cdgan_model()
    new_model.load_state_dict(state_dict)

    # Test that loaded model works
    dummy_input = generate_change_detection_data(batch_size=1, height=64, width=64)

    new_model.eval()
    with torch.no_grad():
        generator_output = new_model.generator(dummy_input)
        discriminator_output = new_model.discriminator(dummy_input, generator_output)

    assert generator_output.shape == (1, 1, 64, 64)
    assert discriminator_output.shape == (1, 1)
