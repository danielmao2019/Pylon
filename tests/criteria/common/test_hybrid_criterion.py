import pytest
import torch
from criteria.wrappers.hybrid_criterion import HybridCriterion
from criteria.vision_2d.dense_prediction.dense_classification.semantic_segmentation import SemanticSegmentationCriterion
from criteria.vision_2d.dense_prediction.dense_classification.dice_loss import DiceLoss


@pytest.fixture
def dummy_data():
    batch_size = 2
    num_classes = 3
    height = 4
    width = 4

    # Create dummy predictions (logits)
    y_pred = torch.randn(batch_size, num_classes, height, width, requires_grad=True)

    # Create dummy ground truth (class indices)
    y_true = torch.randint(0, num_classes, (batch_size, height, width), requires_grad=False)

    return y_pred, y_true


def test_hybrid_criterion_initialization():
    """Test that HybridCriterion can be initialized with valid configurations."""
    criteria_cfg = [
        {
            'class': SemanticSegmentationCriterion,
            'args': {
                'reduction': 'mean',
                'class_weights': None,
                'ignore_value': 255
            }
        },
        {
            'class': DiceLoss,
            'args': {
                'reduction': 'mean',
                'class_weights': None,
                'ignore_value': 255
            }
        }
    ]

    # Test initialization with sum combination
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    assert criterion.combine == 'sum'
    assert len(criterion.criteria) == 2

    # Test initialization with mean combination
    criterion = HybridCriterion(combine='mean', criteria_cfg=criteria_cfg)
    assert criterion.combine == 'mean'
    assert len(criterion.criteria) == 2


def test_hybrid_criterion_forward(dummy_data):
    """Test that HybridCriterion can process dummy data and return a loss value."""
    y_pred, y_true = dummy_data

    criteria_cfg = [
        {
            'class': SemanticSegmentationCriterion,
            'args': {
                'reduction': 'mean',
                'class_weights': None,
                'ignore_value': 255
            }
        },
        {
            'class': DiceLoss,
            'args': {
                'reduction': 'mean',
                'class_weights': None,
                'ignore_value': 255
            }
        }
    ]

    # Test forward pass with sum combination
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    loss = criterion(y_pred=y_pred, y_true=y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar loss
    assert loss.requires_grad  # Loss should require gradients

    # Test forward pass with mean combination
    criterion = HybridCriterion(combine='mean', criteria_cfg=criteria_cfg)
    loss = criterion(y_pred=y_pred, y_true=y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar loss
    assert loss.requires_grad  # Loss should require gradients
