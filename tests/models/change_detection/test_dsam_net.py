import pytest
import torch
from models.change_detection.dsamnet.dsamnet import DSAMNet


@pytest.mark.parametrize("input_size", [(224, 224), (256, 256)])
def test_dsam_net_forward_pass(input_size):
    """Test DSAMNet forward pass with different input sizes"""
    # Create model
    model = DSAMNet(in_c=3, n_class=2).cuda()
    model.eval()

    # Create dummy inputs
    batch_size = 2
    image1 = torch.randn(batch_size, 3, *input_size, device='cuda')
    image2 = torch.randn(batch_size, 3, *input_size, device='cuda')
    inputs = {'image1': image1, 'image2': image2}

    # Run forward pass
    with torch.no_grad():
        output = model(inputs)

    # Verify output shape
    assert output.shape == (batch_size, 2, *input_size), \
        f"Expected output shape (batch_size, 2, *input_size), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaNs"
