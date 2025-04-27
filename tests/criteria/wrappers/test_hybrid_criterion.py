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


def test_buffer_behavior(criteria_cfg, sample_tensor):
    """Test the buffer behavior of HybridCriterion."""
    # Test initialize
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and criterion.buffer == []
    for component_criterion in criterion.criteria:
        assert component_criterion.use_buffer is False
        assert not hasattr(component_criterion, 'buffer')

    # Test update
    loss1 = criterion(y_pred=sample_tensor, y_true=torch.randn_like(sample_tensor))
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and len(criterion.buffer) == 1
    assert criterion.buffer[0].equal(loss1.detach().cpu())
    for component_criterion in criterion.criteria:
        assert component_criterion.use_buffer is False
        assert not hasattr(component_criterion, 'buffer')

    # Test reset
    criterion.reset_buffer()
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and criterion.buffer == []
    for component_criterion in criterion.criteria:
        assert component_criterion.use_buffer is False
        assert not hasattr(component_criterion, 'buffer')


def test_device_transfer(criteria_cfg, sample_tensor):
    """Test moving the criterion between CPU and GPU."""
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create a criterion
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    
    # Move to GPU
    criterion = criterion.cuda()
    
    # Check that the criterion and its component criteria are on GPU
    assert next(criterion.parameters()).is_cuda
    for component_criterion in criterion.criteria:
        assert next(component_criterion.parameters()).is_cuda
    
    # Compute loss on GPU
    gpu_input = sample_tensor.cuda()
    gpu_target = torch.randn_like(gpu_input)
    gpu_loss = criterion(y_pred=gpu_input, y_true=gpu_target)
    
    # Move back to CPU
    criterion = criterion.cpu()
    
    # Check that the criterion and its component criteria are on CPU
    assert not next(criterion.parameters()).is_cuda
    for component_criterion in criterion.criteria:
        assert not next(component_criterion.parameters()).is_cuda
    
    # Compute loss on CPU
    cpu_target = gpu_target.cpu()
    cpu_loss = criterion(y_pred=sample_tensor, y_true=cpu_target)
    
    # Check that the losses are the same
    assert abs(gpu_loss.item() - cpu_loss.item()) < 1e-5
