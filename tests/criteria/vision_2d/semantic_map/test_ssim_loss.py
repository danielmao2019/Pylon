import pytest
from criteria.vision_2d.semantic_map.ssim_loss import SSIMLoss
import torch


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("image_size", [(32, 32), (64, 64)])
def test_ssim_loss(reduction, batch_size, channels, image_size):
    """
    Test SSIMLoss for different reductions, batch sizes, channels, and image sizes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy images
    y_pred = torch.rand(size=(batch_size, channels, *image_size))
    y_true = torch.randint(size=(batch_size, *image_size), low=0, high=channels)

    # Initialize SSIM loss and move to device
    loss_fn = SSIMLoss(window_size=11, channels=channels, reduction=reduction).to(device)
    y_pred = y_pred.to(device)
    y_true = y_true.to(device)

    # Compute loss
    loss = loss_fn(y_pred, y_true)

    # Check output type and shape
    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"
    assert loss.shape == (), f"Expected scalar output but got shape {loss.shape}"
    assert torch.isfinite(loss).all(), "Loss should not contain NaN or Inf values"


@pytest.mark.parametrize("invalid_shape", [(3,), (3, 32), (3, 32, 32)])
def test_invalid_input_shapes(invalid_shape):
    """
    Test that SSIMLoss raises an error for invalid input shapes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize loss and move to device
    loss_fn = SSIMLoss().to(device)

    # Create tensors on device
    img1 = torch.rand(invalid_shape).to(device)
    img2 = torch.rand(invalid_shape).to(device)

    with pytest.raises(ValueError, match="Input images must have shape"):
        loss_fn(img1, img2)


@pytest.mark.parametrize("invalid_channels", [2, 4])
def test_invalid_channel_mismatch(invalid_channels):
    """
    Test that SSIMLoss raises an error if input image channels do not match expected channels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize loss with 3 channels and move to device
    loss_fn = SSIMLoss(window_size=11, channels=3).to(device)

    # Create tensors on device
    img1 = torch.rand((2, invalid_channels, 32, 32)).to(device)
    img2 = torch.rand((2, invalid_channels, 32, 32)).to(device)

    with pytest.raises(ValueError, match=f"Input images must have {3} channels"):
        loss_fn(img1, img2)


def test_window_registered_on_correct_device():
    """
    Test that the Gaussian window is registered on the correct device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = SSIMLoss().to(device)

    assert loss_fn.window.device.type == device.type, \
        f"Window should be on the same device as specified in SSIMLoss. Got {loss_fn.window.device} and {device}."
