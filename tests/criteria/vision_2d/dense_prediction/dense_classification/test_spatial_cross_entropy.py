import pytest
import torch
import torch.nn.functional as F
from criteria.vision_2d.dense_prediction.dense_classification.spatial_cross_entropy import SpatialCrossEntropyCriterion


@pytest.fixture
def sample_data():
    """
    Create sample data for testing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, num_classes = 2, 3
    height, width = 32, 32

    # Create predicted logits with shape (N, C, H, W)
    y_pred = torch.randn(batch_size, num_classes, height, width, device=device)
    # Create ground truth with shape (N, H, W) with values in [0, num_classes)
    y_true = torch.randint(0, num_classes, (batch_size, height, width), device=device)

    return y_pred, y_true


def test_spatial_cross_entropy_basic(sample_data):
    """
    Test basic functionality of SpatialCrossEntropyCriterion.
    """
    y_pred, y_true = sample_data
    device = y_pred.device

    # Initialize criterion
    criterion = SpatialCrossEntropyCriterion(num_classes=y_pred.size(1)).to(device)

    # Compute loss
    loss = criterion(y_pred, y_true)

    # Check output type and shape
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_spatial_cross_entropy_vs_pytorch(sample_data):
    """
    Test that our implementation matches PyTorch's cross entropy when using the same parameters.
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)

    # Initialize our criterion
    criterion = SpatialCrossEntropyCriterion(num_classes=num_classes).to(device)

    # Compute our loss
    our_loss = criterion(y_pred, y_true)

    # Compute PyTorch's loss
    pytorch_loss = F.cross_entropy(y_pred, y_true, reduction='mean')

    # Losses should be close
    assert torch.isclose(our_loss, pytorch_loss, rtol=1e-4).item()


def test_spatial_cross_entropy_with_class_weights(sample_data):
    """
    Test that our implementation matches PyTorch's cross entropy when using class weights.
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)

    # Create unequal class weights
    class_weights = torch.tensor([0.2, 0.3, 0.5], device=device)

    # Initialize our criterion with weights
    criterion = SpatialCrossEntropyCriterion(num_classes=num_classes, class_weights=class_weights).to(device)

    # Compute our loss
    our_loss = criterion(y_pred, y_true)

    # Compute PyTorch's loss with weights
    pytorch_loss = F.cross_entropy(y_pred, y_true, weight=class_weights, reduction='mean')

    # Losses should be close
    assert torch.isclose(our_loss, pytorch_loss, rtol=1e-4).item()


def test_spatial_cross_entropy_with_ignore_index(sample_data):
    """
    Test that our implementation matches PyTorch's cross entropy when using ignore_index.
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)
    ignore_index = 255

    # Create a version with some ignored pixels
    y_true_ignored = y_true.clone()
    y_true_ignored[0, 0, 0] = ignore_index

    # Initialize our criterion with ignore_index
    criterion = SpatialCrossEntropyCriterion(num_classes=num_classes, ignore_index=ignore_index).to(device)

    # Compute our loss
    our_loss = criterion(y_pred, y_true_ignored)

    # Compute PyTorch's loss with ignore_index
    pytorch_loss = F.cross_entropy(y_pred, y_true_ignored, ignore_index=ignore_index, reduction='mean')

    # Losses should be close
    assert torch.isclose(our_loss, pytorch_loss, rtol=1e-4).item()


def test_spatial_cross_entropy_with_weights_and_ignore(sample_data):
    """
    Test that our implementation matches PyTorch's cross entropy when using both class weights and ignore_index.
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)
    ignore_index = 255

    # Create unequal class weights
    class_weights = torch.tensor([0.2, 0.3, 0.5], device=device)

    # Create a version with some ignored pixels
    y_true_ignored = y_true.clone()
    y_true_ignored[0, 0, 0] = ignore_index

    # Initialize our criterion with weights and ignore_index
    criterion = SpatialCrossEntropyCriterion(
        num_classes=num_classes,
        class_weights=class_weights,
        ignore_index=ignore_index
    ).to(device)

    # Compute our loss
    our_loss = criterion(y_pred, y_true_ignored)

    # Compute PyTorch's loss with weights and ignore_index
    pytorch_loss = F.cross_entropy(
        y_pred,
        y_true_ignored,
        weight=class_weights,
        ignore_index=ignore_index,
        reduction='mean'
    )

    # Losses should be close
    assert torch.isclose(our_loss, pytorch_loss, rtol=1e-4).item()


def test_spatial_cross_entropy_all_ignored(sample_data):
    """
    Test that an error is raised when all pixels are ignored.
    """
    y_pred, _ = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)
    ignore_index = 255

    # Create target with all pixels ignored
    y_true = torch.full_like(y_pred[:, 0], fill_value=ignore_index)

    # Initialize criterion
    criterion = SpatialCrossEntropyCriterion(num_classes=num_classes, ignore_index=ignore_index).to(device)

    # Loss computation should raise an error when all pixels are ignored
    with pytest.raises(ValueError, match="All pixels in target are ignored"):
        criterion(y_pred, y_true)
