import pytest
import torch
import numpy as np
from criteria.vision_2d.dense_prediction.dense_regression.instance_segmentation import InstanceSegmentationCriterion


def test_instance_segmentation_init():
    # Test initialization with different ignore indices
    criterion = InstanceSegmentationCriterion(ignore_value=-1)
    assert criterion.ignore_value == -1

    criterion = InstanceSegmentationCriterion(ignore_value=255)
    assert criterion.ignore_value == 255


def test_instance_segmentation_basic():
    criterion = InstanceSegmentationCriterion(ignore_value=-1)
    batch_size = 2
    height = 4
    width = 4

    # Create sample predictions and targets
    y_pred = torch.rand(batch_size, height, width)  # Random predictions
    y_true = torch.randint(0, 10, (batch_size, height, width))  # Random instance IDs

    # Compute loss
    loss = criterion(y_pred, y_true)

    # Check loss properties
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar output
    assert loss.item() >= 0  # L1 loss is always non-negative


def test_instance_segmentation_perfect_predictions():
    criterion = InstanceSegmentationCriterion(ignore_value=-1)
    batch_size = 2
    height = 4
    width = 4

    # Create ground truth instance IDs
    y_true = torch.randint(0, 10, (batch_size, height, width))

    # Set predictions to be exactly the same as ground truth
    y_pred = y_true.float()

    # Compute loss
    loss = criterion(y_pred, y_true)

    # For perfect predictions, L1 loss should be 0
    assert loss.item() < 1e-6


def test_instance_segmentation_with_ignored_regions():
    criterion = InstanceSegmentationCriterion(ignore_value=-1)
    batch_size = 2
    height = 4
    width = 4

    # Create sample data with some ignored regions
    y_true = torch.randint(0, 10, (batch_size, height, width))
    y_true[0, 0, 0] = -1  # Set some pixels to ignore_value
    y_true[1, 1, 1] = -1

    y_pred = torch.rand(batch_size, height, width)

    # Compute loss
    loss = criterion(y_pred, y_true)

    # Check loss properties
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_instance_segmentation_all_ignored():
    criterion = InstanceSegmentationCriterion(ignore_value=-1)
    batch_size = 2
    height = 4
    width = 4

    # Create data with all pixels ignored
    y_true = torch.full((batch_size, height, width), fill_value=-1)
    y_pred = torch.rand(batch_size, height, width)

    # Should raise assertion error when all pixels are ignored
    with pytest.raises(AssertionError):
        criterion(y_pred, y_true)


def test_instance_segmentation_input_validation():
    criterion = InstanceSegmentationCriterion(ignore_value=-1)

    # Test mismatched shapes
    with pytest.raises(AssertionError):
        y_pred = torch.rand(2, 4, 4)
        y_true = torch.randint(0, 10, (2, 4, 5))  # Different width
        criterion(y_pred, y_true)

    # Test mismatched batch size
    with pytest.raises(AssertionError):
        y_pred = torch.rand(2, 4, 4)
        y_true = torch.randint(0, 10, (3, 4, 4))  # Different batch size
        criterion(y_pred, y_true)

    # Test wrong number of dimensions
    with pytest.raises(AssertionError):
        y_pred = torch.rand(2, 1, 4, 4)  # 4D tensor instead of 3D
        y_true = torch.randint(0, 10, (2, 4, 4))
        criterion(y_pred, y_true)
