import pytest
import torch
from models.change_detection.hanet.HANet import HANet


def test_hanet_256():
    """Test HANet forward pass with 256x256 input size"""
    # Create model
    model = HANet(in_ch=3, ou_ch=2)

    # Create dummy inputs
    batch_size = 2
    input_size = (256, 256)
    x1 = torch.randn(batch_size, 3, *input_size)
    x2 = torch.randn(batch_size, 3, *input_size)
    inputs = {'img_1': x1, 'img_2': x2}

    # Run forward pass
    output = model(inputs)
    # Verify output shape
    assert output.shape == (batch_size, 2, *input_size), f"Expected output shape (batch_size, 2, {input_size}), got {output.shape}."


def test_hanet_224():
    """Test HANet forward pass with 224x224 input size should raise size mismatch error"""
    # Create model
    model = HANet(in_ch=3, ou_ch=2)

    # Create dummy inputs
    batch_size = 2
    input_size = (224, 224)
    x1 = torch.randn(batch_size, 3, *input_size)
    x2 = torch.randn(batch_size, 3, *input_size)
    inputs = {'img_1': x1, 'img_2': x2}

    # Expect RuntimeError with size mismatch
    with pytest.raises(RuntimeError) as exc_info:
        model(inputs)
    assert "size" in str(exc_info.value).lower(), "Expected size mismatch error"
