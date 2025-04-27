import pytest
import torch
from criteria.vision_2d.dense_prediction.dense_classification.semantic_segmentation import SemanticSegmentationCriterion


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


def test_semantic_segmentation_basic(sample_data):
    """
    Test basic functionality of SemanticSegmentationCriterion.
    """
    y_pred, y_true = sample_data
    device = y_pred.device

    # Initialize criterion
    criterion = SemanticSegmentationCriterion().to(device)

    # Compute loss
    loss = criterion(y_pred, y_true)

    # Check output type and shape
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_semantic_segmentation_perfect_predictions(sample_data):
    """
    Test SemanticSegmentationCriterion with perfect predictions.
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)

    # Create perfect predictions (one-hot encoded with high confidence)
    y_pred_perfect = torch.zeros_like(y_pred)
    for b in range(y_true.size(0)):
        y_pred_perfect[b].scatter_(0, y_true[b].unsqueeze(0), 100.0)  # High confidence for correct class

    # Initialize criterion
    criterion = SemanticSegmentationCriterion().to(device)

    # Compute loss
    loss = criterion(y_pred_perfect, y_true)

    # Loss should be close to 0 for perfect predictions
    assert loss.item() < 0.1, f"Loss should be close to 0 for perfect predictions, got {loss.item()}"


def test_semantic_segmentation_with_class_weights(sample_data):
    """
    Test SemanticSegmentationCriterion with class weights.
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)

    # Create unequal class weights
    class_weights = torch.tensor([0.2, 0.3, 0.5], device=device)

    # Initialize criterion with weights
    criterion = SemanticSegmentationCriterion(class_weights=class_weights).to(device)
    criterion_no_weights = SemanticSegmentationCriterion().to(device)

    # Compute losses
    loss = criterion(y_pred, y_true)
    loss_no_weights = criterion_no_weights(y_pred, y_true)

    # Check that loss is still valid
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0

    # Loss should be different with unequal weights
    assert not torch.isclose(loss, loss_no_weights, rtol=1e-4).item()


def test_semantic_segmentation_with_ignore_index(sample_data):
    """
    Test SemanticSegmentationCriterion with ignored pixels.
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)
    ignore_value = 255

    # Create a version with some ignored pixels
    y_true_ignored = y_true.clone()
    y_true_ignored[0, 0, 0] = ignore_value

    # Initialize criterion with ignore_value
    criterion = SemanticSegmentationCriterion(ignore_value=ignore_value).to(device)

    # Compute loss
    loss = criterion(y_pred, y_true_ignored)

    # Check that loss is still valid
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_semantic_segmentation_all_ignored(sample_data):
    """
    Test that an error is raised when all pixels are ignored.
    """
    y_pred, _ = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)
    ignore_value = 255

    # Create target with all pixels ignored - ensure it's int64
    y_true = torch.full_like(y_pred[:, 0], fill_value=ignore_value, dtype=torch.int64)

    # Initialize criterion
    criterion = SemanticSegmentationCriterion(ignore_value=ignore_value).to(device)

    # Loss computation should raise an error when all pixels are ignored
    with pytest.raises(ValueError, match="All pixels in target are ignored"):
        criterion(y_pred, y_true)


def test_semantic_segmentation_with_weights_and_ignore(sample_data):
    """
    Test SemanticSegmentationCriterion with both class weights and ignore_value.
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)
    ignore_value = 255

    # Create unequal class weights
    class_weights = torch.tensor([0.2, 0.3, 0.5], device=device)

    # Create a version with some ignored pixels
    y_true_ignored = y_true.clone()
    y_true_ignored[0, 0, 0] = ignore_value

    # Initialize criterion with weights and ignore_value
    criterion = SemanticSegmentationCriterion(
        class_weights=class_weights,
        ignore_value=ignore_value
    ).to(device)

    # Compute loss
    loss = criterion(y_pred, y_true_ignored)

    # Check that loss is still valid
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_semantic_segmentation_input_validation(sample_data):
    """
    Test input validation in SemanticSegmentationCriterion.
    """
    y_pred, y_true = sample_data
    device = y_pred.device
    num_classes = y_pred.size(1)

    # Initialize criterion
    criterion = SemanticSegmentationCriterion().to(device)

    # Test invalid dimensions (not 4D tensor)
    with pytest.raises(AssertionError):
        invalid_pred = torch.randn(3, device=device)  # 1D tensor
        criterion(invalid_pred, y_true)

    # Test mismatched batch size
    with pytest.raises(AssertionError):
        invalid_pred = torch.randn(y_pred.shape[0] + 1, y_pred.shape[1], y_pred.shape[2], y_pred.shape[3], device=device)
        criterion(invalid_pred, y_true)

    # Test out-of-range values in y_true
    y_true_invalid = y_true.clone()
    y_true_invalid[0, 0, 0] = num_classes  # This should be caught by one_hot encoding

    # The assertion happens in the one_hot encoding function
    with pytest.raises(AssertionError, match="Values must be in range"):
        criterion(y_pred, y_true_invalid)
