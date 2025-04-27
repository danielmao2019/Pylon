import pytest
import torch
from criteria.wrappers.spatial_pytorch_criterion_wrapper import SpatialPyTorchCriterionWrapper


@pytest.fixture
def base_criterion():
    """Create a simple criterion for testing."""
    return torch.nn.MSELoss()


@pytest.fixture
def criterion(base_criterion):
    """Create a SpatialPyTorchCriterionWrapper instance for testing."""
    return SpatialPyTorchCriterionWrapper(criterion=base_criterion)


def test_initialization(criterion):
    """Test that the criterion is properly registered as a submodule."""
    # Test that the criterion is properly registered as a submodule
    assert hasattr(criterion, 'criterion')
    assert isinstance(criterion.criterion, torch.nn.MSELoss)

    # Test that the criterion is in the module's children
    assert 'criterion' in dict(criterion.named_children())


def test_compute_loss_same_resolution(criterion, base_criterion, sample_tensor):
    """Test computing loss with inputs of the same resolution."""
    # Create a target tensor with the same resolution
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
    assert len(criterion.buffer) == 1
    assert criterion.buffer[0].equal(loss.detach().cpu())


def test_compute_loss_different_resolution(criterion, base_criterion, sample_tensor):
    """Test computing loss with inputs of different resolutions."""
    # Create a target tensor with different resolution
    y_true = torch.randn(2, 3, 2, 2)

    # Compute loss using the wrapper
    loss = criterion(y_pred=sample_tensor, y_true=y_true)

    # The wrapper should resize y_true to match y_pred's resolution
    # Compute expected loss with resized y_true
    resized_y_true = torch.nn.functional.interpolate(y_true, size=(4, 4), mode='nearest')
    expected_loss = base_criterion(input=sample_tensor, target=resized_y_true)

    # Check that the losses match
    assert loss.item() == expected_loss.item()

    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_buffer_behavior(base_criterion, sample_tensor):
    """Test the buffer behavior of SpatialPyTorchCriterionWrapper."""
    # Test initialize
    criterion = SpatialPyTorchCriterionWrapper(criterion=base_criterion)
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and criterion.buffer == []
    assert criterion.criterion.use_buffer is False
    assert not hasattr(criterion.criterion, 'buffer')
    
    # Test update
    loss1 = criterion(y_pred=sample_tensor, y_true=torch.randn_like(sample_tensor))
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and len(criterion.buffer) == 1
    assert criterion.buffer[0].equal(loss1.detach().cpu())
    assert criterion.criterion.use_buffer is False
    assert not hasattr(criterion.criterion, 'buffer')
    
    # Test reset
    criterion.reset_buffer()
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and criterion.buffer == []
    assert criterion.criterion.use_buffer is False
    assert not hasattr(criterion.criterion, 'buffer')
