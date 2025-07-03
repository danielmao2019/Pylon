from typing import Tuple
import pytest
import torch
from criteria.vision_2d.change_detection.ftn_criterion import FTNCriterion
from criteria.wrappers import AuxiliaryOutputsCriterion
from criteria.vision_2d import SemanticSegmentationCriterion, IoULoss, SSIMLoss


def test_ftn_criterion_initialization():
    """Test that FTNCriterion initializes correctly with buffer management."""
    criterion = FTNCriterion()
    
    # Check that FTNCriterion itself has use_buffer=True (from HybridCriterion/SingleTaskCriterion)
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer')
    
    # Check that it has 3 criteria
    assert len(criterion.criteria) == 3
    
    # Check that all component criteria are AuxiliaryOutputsCriterion with use_buffer=False
    for i, sub_criterion in enumerate(criterion.criteria):
        assert isinstance(sub_criterion, AuxiliaryOutputsCriterion)
        assert sub_criterion.use_buffer is False
        assert not hasattr(sub_criterion, 'buffer')
        
        # Check that the core criteria inside AuxiliaryOutputsCriterion also have use_buffer=False
        assert sub_criterion.criterion.use_buffer is False
        assert not hasattr(sub_criterion.criterion, 'buffer')
    
    # Verify the types of core criteria
    assert isinstance(criterion.criteria[0].criterion, SemanticSegmentationCriterion)
    assert isinstance(criterion.criteria[1].criterion, SSIMLoss)
    assert isinstance(criterion.criteria[2].criterion, IoULoss)


def test_ftn_criterion_forward():
    """Test forward pass of FTNCriterion."""
    batch_size = 2
    num_classes = 2
    height, width = 64, 64
    
    # Create dummy predictions (tuple of 4 tensors for auxiliary outputs)
    y_pred = tuple(
        torch.randn(batch_size, num_classes, height, width, requires_grad=True) 
        for _ in range(4)
    )
    
    # Create dummy ground truth
    y_true = {
        'change_map': torch.randint(0, num_classes, (batch_size, height, width), dtype=torch.int64)
    }
    
    criterion = FTNCriterion()
    
    # Test forward pass
    loss = criterion(y_pred, y_true)
    
    # Check output
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()  # scalar
    assert loss.requires_grad
    assert torch.isfinite(loss)
    
    # Wait for async buffer operations to complete
    criterion._buffer_queue.join()
    
    # Check that loss was added to buffer
    assert len(criterion.buffer) == 1
    assert criterion.buffer[0].shape == ()


def test_ftn_criterion_invalid_inputs():
    """Test that FTNCriterion properly validates inputs."""
    criterion = FTNCriterion()
    batch_size = 2
    height, width = 64, 64
    
    # Test with wrong number of predictions
    with pytest.raises(AssertionError, match="type\\(y_pred\\)=|len\\(y_pred\\)="):
        y_pred = tuple(torch.randn(batch_size, 2, height, width) for _ in range(3))  # 3 instead of 4
        y_true = {'change_map': torch.randint(0, 2, (batch_size, height, width), dtype=torch.int64)}
        criterion(y_pred, y_true)
    
    # Test with wrong type for predictions
    with pytest.raises(AssertionError, match="type\\(y_pred\\)="):
        y_pred = torch.randn(batch_size, 2, height, width)  # tensor instead of tuple
        y_true = {'change_map': torch.randint(0, 2, (batch_size, height, width), dtype=torch.int64)}
        criterion(y_pred, y_true)
    
    # Test with missing 'change_map' key
    with pytest.raises(AssertionError):
        y_pred = tuple(torch.randn(batch_size, 2, height, width) for _ in range(4))
        y_true = {'wrong_key': torch.randint(0, 2, (batch_size, height, width), dtype=torch.int64)}
        criterion(y_pred, y_true)


def test_ftn_criterion_gradient_flow():
    """Test that gradients flow through all components."""
    batch_size = 1
    num_classes = 2
    height, width = 32, 32
    
    # Create predictions that require gradients
    y_pred = tuple(
        torch.randn(batch_size, num_classes, height, width, requires_grad=True) 
        for _ in range(4)
    )
    
    y_true = {
        'change_map': torch.randint(0, num_classes, (batch_size, height, width), dtype=torch.int64)
    }
    
    criterion = FTNCriterion()
    loss = criterion(y_pred, y_true)
    
    # Compute gradients
    loss.backward()
    
    # Check that all predictions have gradients
    for pred in y_pred:
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape
        assert torch.isfinite(pred.grad).all()


def test_ftn_criterion_buffer_accumulation():
    """Test that buffer accumulates losses correctly across multiple calls."""
    criterion = FTNCriterion()
    batch_size = 1
    height, width = 32, 32
    
    # Reset buffer
    criterion.reset_buffer()
    assert len(criterion.buffer) == 0
    
    # Make multiple forward passes
    num_iterations = 5
    for _ in range(num_iterations):
        y_pred = tuple(
            torch.randn(batch_size, 2, height, width) 
            for _ in range(4)
        )
        y_true = {
            'change_map': torch.randint(0, 2, (batch_size, height, width), dtype=torch.int64)
        }
        loss = criterion(y_pred, y_true)
    
    # Wait for async buffer operations to complete
    criterion._buffer_queue.join()
    
    # Check buffer accumulated all losses
    assert len(criterion.buffer) == num_iterations
    for buffered_loss in criterion.buffer:
        assert isinstance(buffered_loss, torch.Tensor)
        assert buffered_loss.shape == ()
