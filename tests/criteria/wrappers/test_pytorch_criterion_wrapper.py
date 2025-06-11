import pytest
import torch
from criteria.base_criterion import BaseCriterion
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper


@pytest.fixture
def base_criterion():
    """Create a simple criterion for testing."""
    return torch.nn.MSELoss()


@pytest.fixture
def criterion(base_criterion):
    """Create a PyTorchCriterionWrapper instance for testing."""
    return PyTorchCriterionWrapper(criterion=base_criterion)


def test_initialization(criterion):
    """Test that the criterion is properly registered as a submodule."""
    # Test that the criterion is properly registered as a submodule
    assert hasattr(criterion, 'criterion')
    assert isinstance(criterion.criterion, torch.nn.MSELoss)

    # Test that the criterion is in the module's children
    assert 'criterion' in dict(criterion.named_children())


def test_compute_loss(criterion, base_criterion, sample_tensor):
    """Test computing loss with the wrapper."""
    # Create a target tensor
    y_true = torch.randn_like(sample_tensor)

    # Compute loss using the wrapper
    loss = criterion(y_pred=sample_tensor, y_true=y_true)

    # Compute loss directly using the base criterion
    expected_loss = base_criterion(input=sample_tensor, target=y_true)

    # Check that the losses match
    assert loss.item() == expected_loss.item()

    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0

    # Check that loss is in the buffer
    criterion._buffer_queue.join()
    assert len(criterion.buffer) == 1
    assert criterion.buffer[0].equal(loss.detach().cpu())


def test_buffer_behavior(base_criterion, sample_tensor):
    """Test the buffer behavior of PyTorchCriterionWrapper."""
    # Test initialize
    criterion = PyTorchCriterionWrapper(criterion=base_criterion)
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and criterion.buffer == []
    assert not isinstance(criterion.criterion, BaseCriterion)

    # Test update
    loss = criterion(y_pred=sample_tensor, y_true=torch.randn_like(sample_tensor))
    criterion._buffer_queue.join()
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and len(criterion.buffer) == 1
    assert criterion.buffer[0].equal(loss.detach().cpu())
    assert not isinstance(criterion.criterion, BaseCriterion)

    # Test reset
    criterion.reset_buffer()
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and criterion.buffer == []
    assert not isinstance(criterion.criterion, BaseCriterion)
