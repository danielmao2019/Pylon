import pytest
import torch
from models.change_detection.dsfernet.dsfernet import DsferNet


def test_dsfernet_valid_input_shape():
    """Test DsferNet with valid input shape (256x256)."""
    # Initialize model
    model = DsferNet(n_classes=2)
    model.eval()
    
    # Create dummy input data
    batch_size = 2
    channels = 3
    input_size = 256
    img_1 = torch.randn(batch_size, channels, input_size, input_size)
    img_2 = torch.randn(batch_size, channels, input_size, input_size)
    
    # Prepare input dictionary
    inputs = {
        'img_1': img_1,
        'img_2': img_2
    }
    
    # Run inference
    with torch.no_grad():
        output = model(inputs)
    
    # Check output structure
    assert isinstance(output, torch.Tensor), "Output should be a tensor in eval mode"
    assert output.shape == (batch_size, 2, input_size, input_size), \
        f"Expected output shape {(batch_size, 2, input_size, input_size)}, got {output.shape}"


def test_dsfernet_invalid_input_shape():
    """Test DsferNet with invalid input shape (224x224) to verify it raises expected error."""
    # Initialize model
    model = DsferNet(n_classes=2)
    model.eval()
    
    # Create dummy input data
    batch_size = 2
    channels = 3
    input_size = 224
    img_1 = torch.randn(batch_size, channels, input_size, input_size)
    img_2 = torch.randn(batch_size, channels, input_size, input_size)
    
    # Prepare input dictionary
    inputs = {
        'img_1': img_1,
        'img_2': img_2
    }
    
    # Run inference and expect RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        with torch.no_grad():
            _ = model(inputs)
    
    # Verify the error message
    assert "Given normalized_shape=[512, 16, 16], expected input with shape [*, 512, 16, 16], but got input of size[2, 512, 14, 14]" in str(exc_info.value), \
        "Expected error about mismatched feature map size"
