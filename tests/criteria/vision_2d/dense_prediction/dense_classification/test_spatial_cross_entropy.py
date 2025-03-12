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
    criterion = SpatialCrossEntropyCriterion().to(device)

    # Compute loss
    loss = criterion(y_pred, y_true)

    # Check output type and shape
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


@pytest.mark.parametrize("use_class_weights,use_ignore_index", [
    (False, False),  # Basic comparison with PyTorch
    (True, False),   # With class weights
    (False, True),   # With ignore_index
    (True, True),    # With both class weights and ignore_index
])
def test_spatial_cross_entropy_vs_pytorch_parametrized(sample_data, use_class_weights, use_ignore_index):
    """
    Test that our implementation matches PyTorch's cross entropy with various parameter combinations.
    
    Args:
        sample_data: Fixture providing input data
        use_class_weights: Whether to test with class weights
        use_ignore_index: Whether to test with ignore_index
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)
    ignore_index = 255 if use_ignore_index else None
    
    # Create class weights if needed
    class_weights = torch.tensor([0.2, 0.3, 0.5], device=device) if use_class_weights else None
    
    # Prepare test data
    if use_ignore_index:
        y_true_modified = y_true.clone()
        y_true_modified[0, 0, 0] = ignore_index  # Set one pixel to ignore_index
    else:
        y_true_modified = y_true
    
    # Initialize our criterion with appropriate parameters
    kwargs = {}
    if class_weights is not None:
        kwargs['class_weights'] = class_weights
    if ignore_index is not None:
        kwargs['ignore_index'] = ignore_index
    
    criterion = SpatialCrossEntropyCriterion(**kwargs).to(device)
    
    # Compute our loss
    our_loss = criterion(y_pred, y_true_modified)
    
    # Compute PyTorch's loss with the same parameters
    pytorch_kwargs = {}
    if class_weights is not None:
        pytorch_kwargs['weight'] = class_weights
    if ignore_index is not None:
        pytorch_kwargs['ignore_index'] = ignore_index
    
    pytorch_loss = F.cross_entropy(y_pred, y_true_modified, reduction='mean', **pytorch_kwargs)
    
    # Losses should be close
    assert torch.isclose(our_loss, pytorch_loss, rtol=1e-4).item(), f"Loss mismatch: ours={our_loss.item()}, pytorch={pytorch_loss.item()}"
    
    # Add descriptive test message based on parameters
    test_description = "Testing SpatialCrossEntropyCriterion "
    if use_class_weights and use_ignore_index:
        test_description += "with both class weights and ignore_index"
    elif use_class_weights:
        test_description += "with class weights"
    elif use_ignore_index:
        test_description += "with ignore_index"
    else:
        test_description += "base functionality"
    
    print(test_description)  # This will show in verbose test output


def test_spatial_cross_entropy_all_ignored(sample_data):
    """
    Test that an error is raised when all pixels are ignored.
    """
    y_pred, _ = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)
    ignore_index = 255

    # Create target with all pixels ignored - ensure it's int64
    y_true = torch.full_like(y_pred[:, 0], fill_value=ignore_index, dtype=torch.int64)

    # Initialize criterion
    criterion = SpatialCrossEntropyCriterion(ignore_index=ignore_index).to(device)

    # Loss computation should raise an error when all pixels are ignored
    with pytest.raises(ValueError, match="All pixels in target are ignored"):
        criterion(y_pred, y_true)
