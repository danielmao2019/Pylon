import torch
import pytest
from models.change_detection.hcgmnet.hcgmnet import HCGMNet


@pytest.mark.parametrize("mode", [
    "train",
    "eval",
])
def test_hcgmnet_forward_pass(mode: str) -> None:
    """Test HCGMNet forward pass in both training and inference modes."""
    model = HCGMNet()
    if mode == "train":
        model.train()
    else:
        model.eval()

    # Create dummy input tensors
    inputs = {
        'img_1': torch.zeros(size=(4, 3, 224, 224)),
        'img_2': torch.zeros(size=(4, 3, 224, 224)),
    }

    # Run forward pass
    outputs = model(inputs)

    if mode == 'train':
        assert isinstance(output, torch.Tensor)
        assert output.shape == (4, 2, 224, 224)
    else:
        assert isinstance(output, dict)
        assert output.keys() == {'intermediate_map', 'change_map'}
        assert output['intermediate_map'].shape == (4, 2, 224, 224)
        assert output['change_map'].shape == (4, 2, 224, 224)


@pytest.mark.parametrize("batch_size,channels,height,width,expected_shape", [
    (2, 3, 256, 256, torch.Size([2, 2, 256, 256])),
    (2, 3, 320, 240, torch.Size([2, 2, 320, 240])),
    (1, 3, 512, 512, torch.Size([1, 2, 512, 512])),
    (4, 3, 128, 128, torch.Size([4, 2, 128, 128]))
])
def test_hcgmnet_different_input_size(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    expected_shape: torch.Size
) -> None:
    """Test HCGMNet with different input sizes."""
    model = HCGMNet(num_classes=2)
    model.eval()

    inputs = {
        'img_1': torch.zeros(size=(batch_size, channels, height, width)),
        'img_2': torch.zeros(size=(batch_size, channels, height, width)),
    }

    outputs = model(inputs)
    assert outputs['change_map'].shape == expected_shape, f"Unexpected shape: {outputs['change_map'].shape}"
