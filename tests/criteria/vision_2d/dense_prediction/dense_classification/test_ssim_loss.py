import pytest
import torch
from criteria.vision_2d.dense_prediction.dense_classification.ssim_loss import SSIMLoss


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
    loss_fn = SSIMLoss(window_size=11, reduction=reduction).to(device)
    y_pred = y_pred.to(device)
    y_true = y_true.to(device)

    # Verify window is on correct device
    assert loss_fn.window.device.type == device.type, \
        f"Window should be on the same device as the model: {loss_fn.window.device} != {device}"

    # Compute loss
    loss = loss_fn(y_pred, y_true)

    # Check output type and shape
    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"
    assert loss.shape == (), f"Expected scalar output but got shape {loss.shape}"
    assert torch.isfinite(loss).all(), "Loss should not contain NaN or Inf values"
    assert loss > 0, "Loss should be positive"


@pytest.mark.parametrize("window_size", [5, 7, 11, 15])
def test_ssim_loss_window_size(window_size):
    """
    Test SSIMLoss with different window sizes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, num_classes = 2, 3
    height, width = 32, 32

    # Create sample data
    y_pred = torch.randn(batch_size, num_classes, height, width).to(device)
    y_true = torch.randint(0, num_classes, (batch_size, height, width)).to(device)

    # Initialize SSIM loss with specified window size
    loss_fn = SSIMLoss(window_size=window_size).to(device)
    
    # Check window dimensions
    assert loss_fn.window.shape == (1, 1, window_size, window_size), \
        f"Window shape should be (1, 1, {window_size}, {window_size}), got {loss_fn.window.shape}"
    
    # Compute loss
    loss = loss_fn(y_pred, y_true)
    
    # Check loss
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()
    assert loss > 0


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


def test_ssim_loss_with_class_weights():
    """
    Test SSIMLoss with class weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, num_classes = 2, 3
    height, width = 32, 32

    # Create sample data
    y_pred = torch.randn(batch_size, num_classes, height, width).to(device)
    y_true = torch.randint(0, num_classes, (batch_size, height, width)).to(device)

    # Test with equal class weights (should be same as no weights)
    loss_fn_equal = SSIMLoss(class_weights=torch.ones(num_classes)).to(device)
    loss_equal = loss_fn_equal(y_pred, y_true)
    
    loss_fn_none = SSIMLoss().to(device)
    loss_none = loss_fn_none(y_pred, y_true)
    
    assert torch.isclose(loss_equal, loss_none, rtol=1e-4).item()
    
    # Test with unequal class weights
    weights = torch.tensor([0.2, 0.3, 0.5]).to(device)
    loss_fn_unequal = SSIMLoss(class_weights=weights).to(device)
    loss_unequal = loss_fn_unequal(y_pred, y_true)
    
    # Loss should be different with unequal weights
    assert not torch.isclose(loss_unequal, loss_none, rtol=1e-4).item()


def test_ssim_loss_with_ignore_index():
    """
    Test SSIMLoss with ignored pixels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, num_classes = 2, 3
    height, width = 32, 32

    # Create sample data
    y_pred = torch.randn(batch_size, num_classes, height, width).to(device)
    y_true = torch.randint(0, num_classes, (batch_size, height, width)).to(device)
    
    # Create a version with some ignored pixels
    y_true_ignored = y_true.clone()
    y_true_ignored[0, 0, 0] = 255  # Set one pixel to ignore_index
    
    loss_fn = SSIMLoss(ignore_index=255).to(device)
    loss = loss_fn(y_pred, y_true_ignored)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_ssim_loss_all_ignored():
    """
    Test SSIMLoss when all pixels are ignored.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, num_classes = 2, 3
    height, width = 32, 32

    # Create predictions and targets where all pixels are ignored
    y_pred = torch.randn(batch_size, num_classes, height, width).to(device)
    y_true = torch.full((batch_size, height, width), fill_value=255).to(device)

    loss_fn = SSIMLoss(ignore_index=255).to(device)
    
    # Loss computation should raise an error when all pixels are ignored
    with pytest.raises(ValueError, match="All pixels in target are ignored"):
        loss_fn(y_pred, y_true)


def test_ssim_loss_with_weights_and_ignore():
    """
    Test SSIMLoss with both class weights and ignored pixels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, num_classes = 2, 3
    height, width = 32, 32

    # Create sample data
    y_pred = torch.randn(batch_size, num_classes, height, width).to(device)
    y_true = torch.randint(0, num_classes, (batch_size, height, width)).to(device)
    
    # Create a version with some ignored pixels
    y_true_ignored = y_true.clone()
    y_true_ignored[0, 0:5, 0:5] = 255  # Set a small block to ignore_index
    
    # Create weights
    weights = torch.tensor([0.2, 0.3, 0.5]).to(device)
    
    # Test with both weights and ignore index
    loss_fn = SSIMLoss(class_weights=weights, ignore_index=255).to(device)
    loss = loss_fn(y_pred, y_true_ignored)
    
    # Test with only weights
    loss_fn_only_weights = SSIMLoss(class_weights=weights).to(device)
    loss_only_weights = loss_fn_only_weights(y_pred, y_true)
    
    # Test with only ignore index
    loss_fn_only_ignore = SSIMLoss(ignore_index=255).to(device)
    loss_only_ignore = loss_fn_only_ignore(y_pred, y_true_ignored)
    
    # The three losses should all be different
    assert not torch.isclose(loss, loss_only_weights, rtol=1e-4).item()
    assert not torch.isclose(loss, loss_only_ignore, rtol=1e-4).item()
    assert not torch.isclose(loss_only_weights, loss_only_ignore, rtol=1e-4).item()


def test_ssim_consistent_across_inputs():
    """
    Test that SSIMLoss produces consistent outputs for similar inputs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, num_classes = 2, 3
    height, width = 32, 32
    
    # Create similar prediction patterns
    y_pred1 = torch.zeros(batch_size, num_classes, height, width).to(device)
    y_pred1[:, 0, :, :] = 1.0  # All pixels predicted as class 0
    
    y_pred2 = torch.zeros(batch_size, num_classes, height, width).to(device)
    y_pred2[:, 1, :, :] = 1.0  # All pixels predicted as class 1
    
    # Create targets where class 0 is correct for y_pred1 and class 1 is correct for y_pred2
    y_true1 = torch.zeros(batch_size, height, width, dtype=torch.long).to(device)
    y_true2 = torch.ones(batch_size, height, width, dtype=torch.long).to(device)
    
    # Initialize loss function
    loss_fn = SSIMLoss().to(device)
    
    # Calculate losses
    loss1 = loss_fn(y_pred1, y_true1)
    loss2 = loss_fn(y_pred2, y_true2)
    
    # Losses should be similar since the patterns are similar, just with different classes
    assert torch.isclose(loss1, loss2, rtol=1e-4).item()
    
    # Cross-computation (incorrect predictions) should yield higher loss
    loss_cross1 = loss_fn(y_pred1, y_true2)
    loss_cross2 = loss_fn(y_pred2, y_true1)
    
    # Cross losses should be higher than correct matches
    assert loss_cross1 > loss1
    assert loss_cross2 > loss2
