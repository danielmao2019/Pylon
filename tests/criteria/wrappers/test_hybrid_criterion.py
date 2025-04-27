import pytest
import torch
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.hybrid_criterion import HybridCriterion


@pytest.fixture
def criteria_cfg():
    """Create criterion configs for testing."""
    return [
        {
            'class': PyTorchCriterionWrapper,
            'args': {
                'criterion': torch.nn.MSELoss(),
            }
        },
        {
            'class': PyTorchCriterionWrapper,
            'args': {
                'criterion': torch.nn.L1Loss(),
            }
        }
    ]


@pytest.fixture
def criterion(criteria_cfg):
    """Create a HybridCriterion instance for testing."""
    return HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)


def test_initialization(criterion):
    """Test that the criteria are properly registered as submodules."""
    # Test that the criteria are properly registered as submodules
    assert hasattr(criterion, 'criteria')
    assert isinstance(criterion.criteria, torch.nn.ModuleList)
    assert len(criterion.criteria) == 2

    # Test that the criteria are in the module's children
    children = dict(criterion.named_children())
    assert 'criteria' in children

    # Test that each criterion is properly registered
    assert isinstance(criterion.criteria[0], PyTorchCriterionWrapper)
    assert isinstance(criterion.criteria[1], PyTorchCriterionWrapper)


def test_compute_loss_sum(criterion, sample_tensor):
    """Test computing loss with sum reduction."""
    # Create a target tensor
    y_true = torch.randn_like(sample_tensor)

    # Compute loss using the wrapper
    loss = criterion(y_pred=sample_tensor, y_true=y_true)

    # Compute expected loss (sum of MSE and L1)
    mse_loss = torch.nn.MSELoss()(input=sample_tensor, target=y_true)
    l1_loss = torch.nn.L1Loss()(input=sample_tensor, target=y_true)
    expected_loss = mse_loss + l1_loss

    # Check that the losses match
    assert loss.item() == expected_loss.item()

    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0

    # Check that loss is in the buffer
    assert len(criterion.buffer) == 1
    assert criterion.buffer[0].equal(loss.detach().cpu())


def test_compute_loss_mean(criteria_cfg, sample_tensor):
    """Test computing loss with mean reduction."""
    # Create a criterion with mean reduction
    criterion_mean = HybridCriterion(combine='mean', criteria_cfg=criteria_cfg)

    # Create a target tensor
    y_true = torch.randn_like(sample_tensor)

    # Compute loss using the wrapper
    loss = criterion_mean(y_pred=sample_tensor, y_true=y_true)

    # Compute expected loss (mean of MSE and L1)
    mse_loss = torch.nn.MSELoss()(input=sample_tensor, target=y_true)
    l1_loss = torch.nn.L1Loss()(input=sample_tensor, target=y_true)
    expected_loss = (mse_loss + l1_loss) / 2

    # Check that the losses match
    assert loss.item() == expected_loss.item()
