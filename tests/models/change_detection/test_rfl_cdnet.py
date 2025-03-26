import pytest
import torch
from models.change_detection.rfl_cdnet.rfl_cdnet import RFL_CDNet


@pytest.mark.parametrize("input_size", [(224, 224), (256, 256)])
def test_rfl_cdnet_forward_pass(input_size):
    """Test RFL_CDNet forward pass with different input sizes"""
    # Create model
    model = RFL_CDNet(in_ch=3, out_ch=2)
    model.eval()

    # Create dummy inputs
    batch_size = 2
    xA = torch.randn(batch_size, 3, *input_size)
    xB = torch.randn(batch_size, 3, *input_size)
    inputs = {'xA': xA, 'xB': xB}

    # Run forward pass
    with torch.no_grad():
        output = model(inputs)

    # Verify output shape
    assert output.shape == (batch_size, 2, *input_size), \
        f"Expected output shape (batch_size, 2, {input_size}), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaNs"
