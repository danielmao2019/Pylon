import pytest
import torch
import numpy as np
from criteria.common.focal_loss import FocalLoss


def test_focal_loss_init():
    # Test default initialization
    loss_fn = FocalLoss()
    assert loss_fn.alpha == 1.0
    assert loss_fn.beta == 2.0

    # Test custom initialization
    loss_fn = FocalLoss(alpha=0.5, beta=3.0)
    assert loss_fn.alpha == 0.5
    assert loss_fn.beta == 3.0

    # Test invalid inputs
    with pytest.raises(AssertionError):
        FocalLoss(alpha="invalid")
    with pytest.raises(AssertionError):
        FocalLoss(beta="invalid")


def test_focal_loss_binary_classification():
    loss_fn = FocalLoss()
    batch_size = 4
    num_classes = 2

    # Create sample predictions and targets
    y_pred = torch.randn(batch_size, num_classes)  # Random logits
    y_true = torch.randint(0, num_classes, (batch_size,))  # Random labels

    # Compute loss
    loss = loss_fn._compute_loss(y_pred, y_true)

    # Check loss properties
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar output
    assert loss.item() > 0  # Loss should be positive


def test_focal_loss_perfect_predictions():
    loss_fn = FocalLoss()
    batch_size = 4

    # Create perfect predictions
    y_pred = torch.tensor([[100.0, -100.0],  # Very confident prediction for class 0
                          [-100.0, 100.0],   # Very confident prediction for class 1
                          [100.0, -100.0],
                          [-100.0, 100.0]])
    y_true = torch.tensor([0, 1, 0, 1])

    loss = loss_fn._compute_loss(y_pred, y_true)
    assert loss.item() < 0.01  # Loss should be very small for perfect predictions


def test_focal_loss_semantic_segmentation():
    loss_fn = FocalLoss()
    batch_size = 2
    height = 4
    width = 4
    num_classes = 2

    # Test 4D input (semantic segmentation)
    y_pred = torch.randn(batch_size, num_classes, height, width)
    y_true = torch.randint(0, num_classes, (batch_size, height, width))

    loss = loss_fn._compute_loss(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_focal_loss_different_alpha_beta():
    # Test how different alpha and beta values affect the loss
    batch_size = 4
    num_classes = 2
    y_pred = torch.tensor([[2.0, -2.0],   # Strong prediction for class 0
                          [-2.0, 2.0],    # Strong prediction for class 1
                          [0.5, -0.5],    # Weak prediction for class 0
                          [-0.5, 0.5]])   # Weak prediction for class 1
    y_true = torch.tensor([1, 1, 0, 0])  # Some correct, some incorrect predictions

    # Compare losses with different alpha values
    loss1 = FocalLoss(alpha=1.0, beta=2.0)._compute_loss(y_pred, y_true)
    loss2 = FocalLoss(alpha=0.5, beta=2.0)._compute_loss(y_pred, y_true)
    assert loss1 != loss2  # Different alpha should give different losses

    # Compare losses with different beta values
    loss3 = FocalLoss(alpha=1.0, beta=3.0)._compute_loss(y_pred, y_true)
    assert loss1 != loss3  # Different beta should give different losses


def test_focal_loss_input_validation():
    loss_fn = FocalLoss()

    # Test invalid number of classes
    with pytest.raises(AssertionError):
        y_pred = torch.randn(4, 3)  # 3 classes instead of 2
        y_true = torch.randint(0, 2, (4,))
        loss_fn._compute_loss(y_pred, y_true)

    # Test mismatched dimensions
    with pytest.raises(Exception):
        y_pred = torch.randn(4, 2)
        y_true = torch.randint(0, 2, (5,))  # Different batch size
        loss_fn._compute_loss(y_pred, y_true)

    # Test invalid target values
    with pytest.raises(Exception):
        y_pred = torch.randn(4, 2)
        y_true = torch.tensor([0, 1, 2, 1])  # Invalid class index 2
        loss_fn._compute_loss(y_pred, y_true)
