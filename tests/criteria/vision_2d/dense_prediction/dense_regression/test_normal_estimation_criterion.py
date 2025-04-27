import pytest
import torch
from criteria.vision_2d.dense_prediction.dense_regression.normal_estimation import NormalEstimationCriterion


def test_normal_estimation_basic():
    criterion = NormalEstimationCriterion()
    batch_size = 2
    height = 4
    width = 4

    # Create sample predictions and targets
    y_pred = torch.randn(batch_size, 3, height, width)  # Random predictions
    y_true = torch.randn(batch_size, 3, height, width)  # Random ground truth
    # Normalize ground truth to make them valid normal vectors
    y_true = y_true / (torch.linalg.norm(y_true, dim=1, keepdim=True) + 1e-6)

    # Compute loss
    loss = criterion(y_pred, y_true)

    # Check loss properties
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar output
    assert 0.0 <= loss.item() <= 2.0  # Cosine similarity loss range (1 - cos_sim)


def test_normal_estimation_perfect_predictions():
    criterion = NormalEstimationCriterion()
    batch_size = 2
    height = 4
    width = 4

    # Create ground truth normals
    y_true = torch.randn(batch_size, 3, height, width)
    # Normalize to make them valid normal vectors
    y_true = y_true / (torch.linalg.norm(y_true, dim=1, keepdim=True) + 1e-6)

    # Set predictions to be exactly the same as ground truth
    y_pred = y_true.clone()

    # Compute loss
    loss = criterion(y_pred, y_true)

    # For perfect predictions, cosine similarity should be 1, so loss should be 0
    assert torch.abs(loss) < 1e-5  # Relax tolerance for floating-point precision


def test_normal_estimation_opposite_predictions():
    criterion = NormalEstimationCriterion()
    batch_size = 2
    height = 4
    width = 4

    # Create ground truth normals
    y_true = torch.randn(batch_size, 3, height, width)
    # Normalize to make them valid normal vectors
    y_true = y_true / (torch.linalg.norm(y_true, dim=1, keepdim=True) + 1e-6)

    # Set predictions to be exactly opposite to ground truth
    y_pred = -y_true.clone()

    # Compute loss
    loss = criterion(y_pred, y_true)

    # For opposite predictions, cosine similarity should be -1, so loss should be 2
    assert torch.abs(loss - 2.0) < 1e-5  # Relax tolerance for floating-point precision


def test_normal_estimation_with_invalid_normals():
    criterion = NormalEstimationCriterion()
    batch_size = 2
    height = 4
    width = 4

    # Create sample data with some invalid normals (zero vectors)
    y_true = torch.randn(batch_size, 3, height, width)
    y_true[:, :, 0, 0] = 0  # Set some normals to zero

    # Normalize the non-zero vectors
    norms = torch.linalg.norm(y_true, dim=1, keepdim=True)
    y_true = torch.where(norms > 0, y_true / (norms + 1e-6), y_true)

    y_pred = torch.randn(batch_size, 3, height, width)

    # Compute loss
    loss = criterion(y_pred, y_true)

    # Check loss properties
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert 0.0 <= loss.item() <= 2.0  # Cosine similarity loss range (1 - cos_sim)


def test_normal_estimation_input_validation():
    criterion = NormalEstimationCriterion()

    # Test invalid number of channels
    with pytest.raises(AssertionError):
        y_pred = torch.randn(2, 4, 4, 4)  # 4 channels instead of 3
        y_true = torch.randn(2, 3, 4, 4)
        criterion(y_pred, y_true)

    # Test mismatched dimensions
    with pytest.raises(AssertionError):
        y_pred = torch.randn(2, 3, 4, 4)
        y_true = torch.randn(2, 3, 4, 5)  # Different width
        criterion(y_pred, y_true)

    # Test invalid input dimensions
    with pytest.raises(AssertionError):
        y_pred = torch.randn(2, 3, 4)  # 3D tensor instead of 4D
        y_true = torch.randn(2, 3, 4, 4)
        criterion(y_pred, y_true)
