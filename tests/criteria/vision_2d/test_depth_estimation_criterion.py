import pytest
import torch
import numpy as np
from criteria.vision_2d.depth_estimation_criterion import DepthEstimationCriterion


def test_depth_estimation_basic():
    criterion = DepthEstimationCriterion()
    batch_size = 2
    height = 4
    width = 4

    # Create sample predictions and targets
    y_pred = torch.rand(batch_size, 1, height, width) * 10  # Random positive depths
    y_true = torch.rand(batch_size, height, width) * 10  # Random positive depths

    # Compute loss
    loss = criterion(y_pred, y_true)

    # Check loss properties
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar output
    assert loss.item() >= 0  # L1 loss is always non-negative


def test_depth_estimation_perfect_predictions():
    criterion = DepthEstimationCriterion()
    batch_size = 2
    height = 4
    width = 4

    # Create ground truth depths
    y_true = torch.rand(batch_size, height, width) * 10  # Random positive depths

    # Set predictions to be exactly the same as ground truth
    y_pred = y_true.unsqueeze(1)  # Add channel dimension

    # Compute loss
    loss = criterion(y_pred, y_true)

    # For perfect predictions, L1 loss should be 0
    assert loss.item() < 1e-6


def test_depth_estimation_with_invalid_depths():
    criterion = DepthEstimationCriterion()
    batch_size = 2
    height = 4
    width = 4

    # Create sample data with some invalid depths (zeros)
    y_true = torch.rand(batch_size, height, width) * 10
    y_true[0, 0, 0] = 0  # Set some depths to zero (invalid)

    y_pred = torch.rand(batch_size, 1, height, width) * 10

    # Compute loss
    loss = criterion(y_pred, y_true)

    # Check loss properties
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_depth_estimation_all_invalid():
    criterion = DepthEstimationCriterion()
    batch_size = 2
    height = 4
    width = 4

    # Create data with all invalid depths
    y_true = torch.zeros(batch_size, height, width)
    y_pred = torch.rand(batch_size, 1, height, width)

    # Should raise assertion error when all depths are invalid
    with pytest.raises(AssertionError):
        criterion(y_pred, y_true)


def test_depth_estimation_input_validation():
    criterion = DepthEstimationCriterion()

    # Test invalid number of channels in prediction
    with pytest.raises(AssertionError):
        y_pred = torch.randn(2, 2, 4, 4)  # 2 channels instead of 1
        y_true = torch.randn(2, 4, 4)
        criterion(y_pred, y_true)

    # Test mismatched dimensions
    with pytest.raises(AssertionError):
        y_pred = torch.randn(2, 1, 4, 4)
        y_true = torch.randn(2, 4, 5)  # Different width
        criterion(y_pred, y_true)

    # Test negative depths in ground truth
    with pytest.raises(AssertionError):
        y_pred = torch.rand(2, 1, 4, 4)  # Positive values
        y_true = -torch.rand(2, 4, 4)  # Negative values
        criterion(y_pred, y_true)
