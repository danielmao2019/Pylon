import pytest
import torch
from criteria.vision_2d.semantic_map.ssim_loss import SSIMLoss


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_classes", [2, 3, 4])
@pytest.mark.parametrize("image_size", [(32, 32), (64, 64)])
def test_ssim_loss(reduction, batch_size, num_classes, image_size):
    """
    Test SSIMLoss for different reductions, batch sizes, number of classes, and image sizes.
    Also verifies device handling and window creation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create predicted logits with shape (N, C, H, W)
    y_pred = torch.randn(size=(batch_size, num_classes, *image_size))
    # Create ground truth with shape (N, H, W) with values in [0, num_classes)
    y_true = torch.randint(0, num_classes, (batch_size, *image_size))

    # Initialize SSIM loss and move to device
    loss_fn = SSIMLoss(num_classes=num_classes, window_size=11, reduction=reduction).to(device)
    y_pred = y_pred.to(device)
    y_true = y_true.to(device)

    # Verify window is on correct device
    assert loss_fn.window.device == device, "Window should be on the same device as the model"

    # Compute loss
    loss = loss_fn(y_pred, y_true)

    # Check output type and shape
    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"
    assert loss.shape == (), f"Expected scalar output but got shape {loss.shape}"
    assert torch.isfinite(loss).all(), "Loss should not contain NaN or Inf values"
    assert loss > 0, "Loss should be positive"


@pytest.mark.parametrize("invalid_shape", [(3,), (3, 32), (3, 32, 32)])
def test_invalid_input_shapes(invalid_shape):
    """
    Test that SSIMLoss raises an error for invalid input shapes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize SSIM loss and move to device
    loss_fn = SSIMLoss(num_classes=2).to(device)

    # Create tensors with invalid shapes
    y_pred = torch.randn(invalid_shape).to(device)
    y_true = torch.randint(0, 2, invalid_shape).to(device)

    with pytest.raises(AssertionError):
        loss_fn(y_pred, y_true)
