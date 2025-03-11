import pytest
import torch
from criteria.vision_2d.semantic_map.spatial_cross_entropy_criterion import SpatialCrossEntropyCriterion


def test_spatial_cross_entropy_init():
    # Test initialization with default settings
    criterion = SpatialCrossEntropyCriterion(ignore_index=255)
    assert criterion.ignore_index == 255
    assert criterion.class_weights is None

    # Test initialization with class weights
    class_weights = (0.2, 0.3, 0.5)  # Must sum to 1 after normalization
    criterion = SpatialCrossEntropyCriterion(ignore_index=255, class_weights=class_weights)
    assert criterion.ignore_index == 255
    assert torch.allclose(criterion.class_weights, torch.tensor(class_weights))

    # Test invalid class weights
    with pytest.raises(AssertionError):
        SpatialCrossEntropyCriterion(ignore_index=255, class_weights=(-0.1, 0.5, 0.6))  # Negative weight

    with pytest.raises(AssertionError):
        SpatialCrossEntropyCriterion(ignore_index=255, class_weights=[0.3, 0.7])  # List instead of tuple


def test_spatial_cross_entropy_basic():
    criterion = SpatialCrossEntropyCriterion(ignore_index=255)
    batch_size, num_classes = 2, 4
    height, width = 32, 32

    # Create sample predictions and targets
    y_pred = torch.randn(batch_size, num_classes, height, width)  # Random logits
    y_true = torch.randint(0, num_classes, (batch_size, height, width))  # Random labels

    # Compute loss
    loss = criterion(y_pred, y_true)

    # Check loss properties
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar output
    assert loss.item() > 0  # Cross entropy loss is always positive


def test_spatial_cross_entropy_perfect_predictions():
    criterion = SpatialCrossEntropyCriterion(ignore_index=255)
    batch_size, num_classes = 2, 4
    height, width = 32, 32

    # Create ground truth labels
    y_true = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Create perfect predictions (one-hot encoded with high confidence)
    y_pred = torch.zeros(batch_size, num_classes, height, width)
    for b in range(batch_size):
        y_pred[b].scatter_(0, y_true[b].unsqueeze(0), 100.0)  # High confidence for correct class

    # Compute loss
    loss = criterion(y_pred, y_true)

    # For perfect predictions with high confidence, loss should be very small
    assert loss.item() < 0.01


def test_spatial_cross_entropy_with_ignored_regions():
    criterion = SpatialCrossEntropyCriterion(ignore_index=255)
    batch_size, num_classes = 2, 4
    height, width = 32, 32

    # Create sample data with some ignored regions
    y_true = torch.randint(0, num_classes, (batch_size, height, width))
    y_true[:, :5, :5] = 255  # Set top-left corner to ignore_index
    
    y_pred = torch.randn(batch_size, num_classes, height, width)

    # Compute loss
    loss = criterion(y_pred, y_true)

    # Check loss properties
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_spatial_cross_entropy_all_ignored():
    criterion = SpatialCrossEntropyCriterion(ignore_index=255)
    batch_size, num_classes = 2, 4
    height, width = 32, 32

    # Create predictions and targets where all pixels are ignored
    y_pred = torch.randn(batch_size, num_classes, height, width)
    y_true = torch.full((batch_size, height, width), fill_value=255)

    # Loss computation should raise an error when all pixels are ignored
    with pytest.raises(ValueError, match="All pixels in target are ignored"):
        criterion(y_pred, y_true)


def test_spatial_cross_entropy_resolution_mismatch():
    criterion = SpatialCrossEntropyCriterion(ignore_index=255)
    batch_size, num_classes = 2, 4
    
    # Create predictions and targets with different resolutions
    y_pred = torch.randn(batch_size, num_classes, 64, 64)  # Higher resolution
    y_true = torch.randint(0, num_classes, (batch_size, 32, 32))  # Lower resolution

    # Loss should be computed after resizing predictions to match target
    loss = criterion(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_spatial_cross_entropy_dynamic_class_weights():
    criterion = SpatialCrossEntropyCriterion(ignore_index=255)
    batch_size, num_classes = 2, 4
    height, width = 32, 32

    # Create imbalanced dataset where one class is rare
    y_true = torch.zeros((batch_size, height, width), dtype=torch.int64)
    y_true[:, :2, :2] = 1  # Only a few pixels of class 1
    
    y_pred = torch.randn(batch_size, num_classes, height, width)

    # First loss computation with dynamic weights
    loss1 = criterion(y_pred, y_true)
    
    # The rare class should have received a higher weight
    assert criterion.temp_weights[1] > criterion.temp_weights[0]

    # Second loss computation should use the same weights
    loss2 = criterion(y_pred, y_true)
    assert torch.allclose(loss1, loss2)


def test_spatial_cross_entropy_with_class_weights():
    batch_size, num_classes = 2, 3
    height, width = 32, 32
    class_weights = (0.1, 0.1, 0.8)  # Much higher weight for the last class
    criterion = SpatialCrossEntropyCriterion(ignore_index=255, class_weights=class_weights)

    # Create two batches of predictions with errors in different classes
    y_true_1 = torch.full((batch_size, height, width), fill_value=2)  # All pixels belong to last class
    y_true_2 = torch.full((batch_size, height, width), fill_value=0)  # All pixels belong to first class
    
    # Same wrong predictions for both cases
    y_pred = torch.zeros(batch_size, num_classes, height, width)
    y_pred[:, 0] = 5.0  # Strong prediction for first class
    y_pred[:, 1] = 0.0  # Neutral for second class
    y_pred[:, 2] = -5.0  # Strong wrong prediction for third class

    # Loss when wrong on the highly weighted class
    loss_high_weight = criterion(y_pred, y_true_1)
    
    # Loss when wrong on the lower weighted class
    loss_low_weight = criterion(y_pred, y_true_2)

    # Loss should be higher when we're wrong on the highly weighted class
    assert loss_high_weight > loss_low_weight
    # The difference should be significant
    assert (loss_high_weight - loss_low_weight) > 1.0


def test_spatial_cross_entropy_input_validation():
    criterion = SpatialCrossEntropyCriterion(ignore_index=255)

    # Test invalid number of dimensions in predictions
    with pytest.raises(AssertionError):
        y_pred = torch.randn(2, 4, 32, 32, 1)  # Extra dimension
        y_true = torch.randint(0, 4, (2, 32, 32))
        criterion(y_pred, y_true)

    # Test invalid number of dimensions in targets
    with pytest.raises(AssertionError):
        y_pred = torch.randn(2, 4, 32, 32)
        y_true = torch.randint(0, 4, (2, 32, 32, 1))  # Extra dimension
        criterion(y_pred, y_true)

    # Test mismatched batch sizes
    with pytest.raises(AssertionError):
        y_pred = torch.randn(2, 4, 32, 32)
        y_true = torch.randint(0, 4, (3, 32, 32))  # Different batch size
        criterion(y_pred, y_true)
