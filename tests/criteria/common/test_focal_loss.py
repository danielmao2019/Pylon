import pytest
import torch
from criteria.common.focal_loss import FocalLoss
from criteria.vision_2d.dense_prediction.dense_classification.semantic_segmentation import SemanticSegmentationCriterion


@pytest.fixture
def sample_data():
    """
    Create sample data for testing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, num_classes = 4, 3
    height, width = 32, 32

    # Create predicted logits with shape (N, C, H, W) and enable gradients
    y_pred = torch.randn(batch_size, num_classes, height, width, device=device, requires_grad=True)
    # Create ground truth with shape (N, H, W) with values in [0, num_classes) and disable gradients
    y_true = torch.randint(0, num_classes, (batch_size, height, width), device=device, requires_grad=False)

    return y_pred, y_true


def test_focal_loss_initialization():
    """Test initialization of FocalLoss with different parameters."""
    # Test default initialization
    loss_fn = FocalLoss()
    assert loss_fn.class_weights is None
    assert loss_fn.gamma == 0.0

    # Test custom initialization
    class_weights = torch.tensor([0.5, 0.5])
    loss_fn = FocalLoss(class_weights=class_weights, gamma=3.0)
    assert torch.equal(loss_fn.class_weights, class_weights)
    assert loss_fn.gamma == 3.0

    # Test invalid inputs
    with pytest.raises(AssertionError):
        FocalLoss(gamma="invalid")
    with pytest.raises(AssertionError):
        FocalLoss(class_weights=torch.randn(2, 2))  # 2D tensor instead of 1D


@pytest.mark.parametrize("input_shape", [
    (4, 2),  # 2D input (classification)
    (4, 2, 8),  # 3D input (single image)
    (2, 2, 4, 4),  # 4D input (batched images)
])
def test_focal_loss_input_shapes(input_shape):
    """
    Test FocalLoss with different input shapes.
    Args:
        input_shape (tuple): Shape of the input tensor (N, C, ...)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create predicted logits with the given shape and enable gradients
    y_pred = torch.randn(*input_shape, device=device, requires_grad=True)

    # Create ground truth with appropriate shape (remove C dimension)
    y_true_shape = (input_shape[0],) + input_shape[2:]
    y_true = torch.randint(0, 2, y_true_shape, device=device, requires_grad=False)

    # Initialize criterion
    criterion = FocalLoss().to(device)

    # Compute loss
    loss = criterion(y_pred, y_true)

    # Check output type and shape
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar loss
    assert loss.item() > 0
    assert loss.requires_grad  # Loss should require gradients

    # Test backward pass
    loss.backward()
    assert y_pred.grad is not None
    assert y_pred.grad.shape == y_pred.shape


def test_focal_loss_perfect_predictions(sample_data):
    """Test FocalLoss with perfect predictions."""
    loss_fn = FocalLoss()

    # Test binary classification case
    y_pred = torch.tensor([[100.0, -100.0],  # Very confident prediction for class 0
                          [-100.0, 100.0],   # Very confident prediction for class 1
                          [100.0, -100.0],
                          [-100.0, 100.0]])
    y_true = torch.tensor([0, 1, 0, 1])
    loss = loss_fn(y_pred=y_pred, y_true=y_true)
    assert loss.item() < 0.01  # Loss should be very small for perfect predictions

    # Test semantic segmentation case
    y_pred, y_true = sample_data
    y_pred_perfect = torch.zeros_like(y_pred)
    for b in range(y_true.size(0)):
        y_pred_perfect[b].scatter_(0, y_true[b].unsqueeze(0), 100.0)
    loss = loss_fn(y_pred=y_pred_perfect, y_true=y_true)
    assert loss.item() < 0.1


def test_focal_loss_class_weights_effect():
    """Test how different class weights and gamma values affect the loss."""
    y_pred = torch.tensor([[2.0, -2.0],   # Strong prediction for class 0
                          [-2.0, 2.0],    # Strong prediction for class 1
                          [0.5, -0.5],    # Weak prediction for class 0
                          [-0.5, 0.5]])   # Weak prediction for class 1
    y_true = torch.tensor([1, 1, 0, 0])  # Some correct, some incorrect predictions

    # Test class weights effect
    loss1 = FocalLoss(class_weights=torch.tensor([1.0, 1.0]), gamma=2.0)(y_pred=y_pred, y_true=y_true)
    loss2 = FocalLoss(class_weights=torch.tensor([0.5, 0.5]), gamma=2.0)(y_pred=y_pred, y_true=y_true)
    loss3 = FocalLoss(class_weights=torch.tensor([2.0, 2.0]), gamma=2.0)(y_pred=y_pred, y_true=y_true)
    assert torch.isclose(loss1, 2 * loss2, rtol=1e-6)  # weights=[1,1] should be 2x weights=[0.5,0.5]
    assert torch.isclose(loss3, 4 * loss2, rtol=1e-6)  # weights=[2,2] should be 4x weights=[0.5,0.5]


def test_focal_loss_gamma_effect():
    y_pred = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
    y_true = torch.tensor([0, 1])
    loss_2 = FocalLoss(gamma=2.0)(y_pred=y_pred, y_true=y_true)
    loss_3 = FocalLoss(gamma=3.0)(y_pred=y_pred, y_true=y_true)
    assert loss_3 < loss_2  # Higher gamma should give lower loss for hard examples


def test_focal_loss_matches_semantic_segmentation(sample_data):
    """Test that FocalLoss matches SemanticSegmentationCriterion with same parameters."""
    y_pred, y_true = sample_data
    device = y_pred.device

    # Initialize both criteria with the same parameters
    focal_criterion = FocalLoss().to(device)
    semantic_criterion = SemanticSegmentationCriterion().to(device)

    # Test regular case
    focal_loss = focal_criterion(y_pred=y_pred, y_true=y_true)
    semantic_loss = semantic_criterion(y_pred=y_pred, y_true=y_true)
    assert torch.isclose(focal_loss, semantic_loss, rtol=1e-6, atol=1e-6)


def test_focal_loss_input_validation():
    """Test input validation in FocalLoss."""
    loss_fn = FocalLoss()

    # Test mismatched dimensions
    with pytest.raises(Exception):
        y_pred = torch.randn(4, 2)
        y_true = torch.randint(0, 2, (5,))  # Different batch size
        loss_fn(y_pred=y_pred, y_true=y_true)

    # Test invalid target values
    with pytest.raises(Exception):
        y_pred = torch.randn(4, 2)
        y_true = torch.tensor([0, 1, 2, 1])  # Invalid class index 2
        loss_fn(y_pred=y_pred, y_true=y_true)

    # Test class weights dimension mismatch
    with pytest.raises(AssertionError):
        y_pred = torch.randn(4, 2)  # 2 classes
        y_true = torch.tensor([0, 1, 0, 1])
        class_weights = torch.tensor([1.0, 1.0, 1.0])  # 3 weights for 2 classes
        loss_fn = FocalLoss(class_weights=class_weights)
        loss_fn(y_pred=y_pred, y_true=y_true)


def test_focal_loss_ignore_value():
    """Test FocalLoss with ignore_value parameter."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create sample data with shape (N, C, H, W)
    batch_size, num_classes = 2, 3
    height, width = 4, 4
    y_pred = torch.randn(batch_size, num_classes, height, width, device=device, requires_grad=True)
    y_true = torch.randint(0, num_classes, (batch_size, height, width), device=device, requires_grad=False)

    # Create a mask with some ignored values
    ignore_mask = torch.zeros_like(y_true, dtype=torch.bool)
    ignore_mask[0, 1:3, 1:3] = True  # Ignore a 2x2 region in the first image
    y_true[ignore_mask] = -1  # Set ignored values to -1

    # Initialize criterion with default ignore_value (-1)
    criterion = FocalLoss().to(device)

    # Compute loss
    loss = criterion(y_pred=y_pred, y_true=y_true)

    # Test with custom ignore value
    custom_ignore_value = 255
    y_true_custom = y_true.clone()
    y_true_custom[ignore_mask] = custom_ignore_value
    criterion_custom = FocalLoss(ignore_value=custom_ignore_value).to(device)

    # Compute loss with custom ignore value
    loss_custom = criterion_custom(y_pred=y_pred, y_true=y_true_custom)

    # Verify that losses are equal (since we're ignoring the same pixels)
    assert torch.isclose(loss, loss_custom, rtol=1e-6, atol=1e-6)

    # Test that ignored values don't affect the loss
    y_pred_modified = y_pred.clone()
    # Broadcast ignore_mask to match y_pred's shape (N, C, H, W)
    ignore_mask_broadcasted = ignore_mask.unsqueeze(1).expand_as(y_pred)
    # Fill ignored locations with random noise
    y_pred_modified[ignore_mask_broadcasted] = torch.randn_like(y_pred_modified[ignore_mask_broadcasted])
    loss_modified = criterion(y_pred=y_pred_modified, y_true=y_true)
    assert torch.isclose(loss, loss_modified, rtol=1e-6, atol=1e-6)
