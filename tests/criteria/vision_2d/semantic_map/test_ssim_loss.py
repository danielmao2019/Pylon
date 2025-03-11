import pytest
import torch
from criteria.vision_2d.semantic_map.ssim_loss import SSIMLoss


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_classes", [2, 3])
@pytest.mark.parametrize("image_size", [(32, 32), (64, 64)])
def test_ssim_loss(reduction, batch_size, num_classes, image_size):
    """
    Test SSIMLoss for different reductions, batch sizes, number of classes, and image sizes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create predicted logits with shape (N, C, H, W)
    y_pred = torch.randn(size=(batch_size, num_classes, *image_size))
    # Create ground truth with shape (N, H, W) with values in [0, num_classes)
    y_true = torch.randint(0, num_classes, (batch_size, *image_size))

    # Initialize SSIM loss and move to device
    loss_fn = SSIMLoss(window_size=11, reduction=reduction).to(device)
    y_pred = y_pred.to(device)
    y_true = y_true.to(device)

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
    loss_fn = SSIMLoss().to(device)

    # Create tensors with invalid shapes
    y_pred = torch.randn(invalid_shape).to(device)
    y_true = torch.randint(0, 2, invalid_shape).to(device)

    with pytest.raises(AssertionError):
        loss_fn(y_pred, y_true)


@pytest.mark.parametrize("num_classes", [2, 4])
def test_class_mismatch(num_classes):
    """
    Test that SSIMLoss works correctly with different numbers of classes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize SSIM loss and move to device
    loss_fn = SSIMLoss(window_size=11).to(device)

    # Create tensors with num_classes channels
    y_pred = torch.randn((2, num_classes, 32, 32)).to(device)
    # Create ground truth with values in [0, num_classes)
    y_true = torch.randint(0, num_classes, (2, 32, 32)).to(device)

    # Compute loss
    loss = loss_fn(y_pred, y_true)
    assert torch.isfinite(loss).all(), "Loss should not contain NaN or Inf values"
    assert loss > 0, "Loss should be positive"


def test_window_device():
    """
    Test that the Gaussian window is created on the correct device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize SSIM loss and move to device
    loss_fn = SSIMLoss().to(device)

    # Create dummy input
    y_pred = torch.randn((1, 3, 32, 32)).to(device)
    y_true = torch.randint(0, 3, (1, 32, 32)).to(device)

    # Compute loss to ensure window is created
    loss = loss_fn(y_pred, y_true)
    assert torch.isfinite(loss).all(), "Loss should not contain NaN or Inf values"
    assert loss > 0, "Loss should be positive"
