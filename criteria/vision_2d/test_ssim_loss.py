import pytest
from .ssim_loss import SSIMLoss
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
    img1 = torch.rand((batch_size, channels, *image_size), device=device)
    img2 = torch.rand((batch_size, channels, *image_size), device=device)

    # Initialize SSIM loss
    loss_fn = SSIMLoss(window_size=11, channels=channels, reduction=reduction, device=device)

    # Compute loss
    loss = loss_fn(img1, img2)

    # Check output type and shape
    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"

    assert loss.shape == (), f"Expected scalar output but got shape {loss.shape}"

    # Check values are finite
    assert torch.isfinite(loss).all(), "Loss should not contain NaN or Inf values"


@pytest.mark.parametrize("invalid_shape", [(3,), (3, 32), (3, 32, 32)])
def test_invalid_input_shapes(invalid_shape):
    """
    Test that SSIMLoss raises an error for invalid input shapes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = SSIMLoss()

    img1 = torch.rand(invalid_shape, device=device)
    img2 = torch.rand(invalid_shape, device=device)

    with pytest.raises(ValueError, match="Input images must have shape"):
        loss_fn(img1, img2)


@pytest.mark.parametrize("invalid_channels", [2, 4])
def test_invalid_channel_mismatch(invalid_channels):
    """
    Test that SSIMLoss raises an error if input image channels do not match expected channels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = SSIMLoss(window_size=11, channels=3, device=device)  # Expecting 3 channels

    img1 = torch.rand((2, invalid_channels, 32, 32), device=device)
    img2 = torch.rand((2, invalid_channels, 32, 32), device=device)

    with pytest.raises(ValueError, match=f"Input images must have {3} channels"):
        loss_fn(img1, img2)


def test_window_registered_on_correct_device():
    """
    Test that the Gaussian window is registered on the correct device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = SSIMLoss(device=device)

    assert loss_fn.window.device == device, "Window should be on the same device as specified in SSIMLoss"
