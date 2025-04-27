import pytest
import torch
from criteria.wrappers.auxiliary_outputs_criterion import AuxiliaryOutputsCriterion


@pytest.fixture
def criterion_cfg():
    """Create a simple criterion config for testing."""
    return {
        'class': torch.nn.MSELoss,
        'args': {}
    }


@pytest.fixture
def criterion(criterion_cfg):
    """Create an AuxiliaryOutputsCriterion instance for testing."""
    return AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)


def test_initialization(criterion):
    """Test that the criterion is properly registered as a submodule."""
    # Test that the criterion is properly registered as a submodule
    assert hasattr(criterion, 'criterion')
    assert isinstance(criterion.criterion, torch.nn.MSELoss)

    # Test that the criterion is in the module's children
    assert 'criterion' in dict(criterion.named_children())


def test_call_with_list_input(criterion, sample_tensors, sample_tensor):
    """Test calling the criterion with a list of predictions."""
    # Compute loss
    loss = criterion(y_pred=sample_tensors, y_true=sample_tensor)

    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0

    # Check that loss is in the buffer
    assert len(criterion.buffer) == 1
    assert criterion.buffer[0].equal(loss.detach().cpu())


def test_call_with_dict_input(criterion, sample_tensor_dict, sample_tensor):
    """Test calling the criterion with a dictionary of predictions."""
    # Compute loss
    loss = criterion(y_pred=sample_tensor_dict, y_true=sample_tensor)

    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_reduction_options(criterion_cfg, sample_tensors, sample_tensor):
    """Test different reduction options."""
    # Test with 'mean' reduction
    criterion_mean = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg, reduction='mean')
    loss_mean = criterion_mean(y_pred=sample_tensors, y_true=sample_tensor)

    # Test with 'sum' reduction
    criterion_sum = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg, reduction='sum')
    loss_sum = criterion_sum(y_pred=sample_tensors, y_true=sample_tensor)

    # The mean loss should be half of the sum loss
    assert abs(loss_mean.item() - loss_sum.item() / 2) < 1e-5
