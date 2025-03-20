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
        assert isinstance(outputs, dict), f"{type(outputs)=}"
        assert outputs.keys() == {'intermediate_map', 'change_map'}, f"{outputs.keys()=}"
        assert outputs['intermediate_map'].shape == (4, 2, 224, 224), f"{outputs['intermediate_map'].shape=}"
        assert outputs['change_map'].shape == (4, 2, 224, 224), f"{outputs['change_map'].shape=}"
    else:
        assert isinstance(outputs, torch.Tensor), f"{type(outputs)=}"
        assert outputs.shape == (4, 2, 224, 224), f"{outputs.shape=}"


@pytest.mark.parametrize("spatial_dim", [
    112,
    128,
    224,
    256,
])
def test_hcgmnet_different_input_size(spatial_dim: int) -> None:
    """Test HCGMNet with different spatial dimensions."""
    model = HCGMNet()
    model.eval()

    inputs = {
        'img_1': torch.zeros(size=(4, 3, spatial_dim, spatial_dim)),
        'img_2': torch.zeros(size=(4, 3, spatial_dim, spatial_dim)),
    }

    outputs = model(inputs)
    assert isinstance(outputs, torch.Tensor), "Model output should be a tensor in eval mode"
    assert outputs.shape == (4, 2, spatial_dim, spatial_dim), f"{outputs.shape=}"
