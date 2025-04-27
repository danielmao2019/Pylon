import pytest
import torch
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.hybrid_criterion import HybridCriterion


@pytest.fixture
def criteria_cfg(dummy_criterion):
    """Create criterion configs with criteria that have registered buffers."""
    return [
        {
            'class': PyTorchCriterionWrapper,
            'args': {
                'criterion': dummy_criterion,
            }
        },
        {
            'class': PyTorchCriterionWrapper,
            'args': {
                'criterion': dummy_criterion,
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

    # Compute expected loss (sum of both criteria)
    expected_loss = sum(c(sample_tensor, y_true) for c in criterion.criteria)

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

    # Compute expected loss (mean of both criteria)
    expected_loss = sum(c(sample_tensor, y_true) for c in criterion_mean.criteria) / 2

    # Check that the losses match
    assert loss.item() == expected_loss.item()


def test_buffer_behavior(criteria_cfg, sample_tensor):
    """Test the buffer behavior of HybridCriterion."""
    # Create a criterion
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    
    # Test initialize
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and criterion.buffer == []
    for component_criterion in criterion.criteria:
        assert component_criterion.use_buffer is False
        assert not hasattr(component_criterion, 'buffer')
    
    # Test update
    y_true = torch.randn_like(sample_tensor)
    loss1 = criterion(y_pred=sample_tensor, y_true=y_true)
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
    
    # Step 1: Test on CPU
    # Check initial state
    assert not next(criterion.parameters()).is_cuda
    for component_criterion in criterion.criteria:
        assert not next(component_criterion.parameters()).is_cuda
        assert not component_criterion.criterion.class_weights.is_cuda
    assert len(criterion.buffer) == 0
    
    # Compute loss on CPU
    y_true = torch.randn_like(sample_tensor)
    cpu_loss = criterion(y_pred=sample_tensor, y_true=y_true)
    assert len(criterion.buffer) == 1
    
    # Step 2: Move to GPU
    criterion = criterion.cuda()
    gpu_input = sample_tensor.cuda()
    gpu_target = y_true.cuda()
    
    # Check GPU state
    assert next(criterion.parameters()).is_cuda
    for component_criterion in criterion.criteria:
        assert next(component_criterion.parameters()).is_cuda
        assert component_criterion.criterion.class_weights.is_cuda
    assert len(criterion.buffer) == 1
    
    # Compute loss on GPU
    gpu_loss = criterion(y_pred=gpu_input, y_true=gpu_target)
    assert len(criterion.buffer) == 2
    
    # Step 3: Move back to CPU
    criterion = criterion.cpu()
    
    # Check CPU state
    assert not next(criterion.parameters()).is_cuda
    for component_criterion in criterion.criteria:
        assert not next(component_criterion.parameters()).is_cuda
        assert not component_criterion.criterion.class_weights.is_cuda
    assert len(criterion.buffer) == 2
    
    # Compute loss on CPU again
    cpu_loss2 = criterion(y_pred=sample_tensor, y_true=y_true)
    assert len(criterion.buffer) == 3
    
    # Check that all losses are equivalent
    assert abs(cpu_loss.item() - gpu_loss.item()) < 1e-5
    assert abs(cpu_loss.item() - cpu_loss2.item()) < 1e-5
