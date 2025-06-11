import pytest
import torch
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.auxiliary_outputs_criterion import AuxiliaryOutputsCriterion


@pytest.fixture
def criterion_cfg(dummy_criterion):
    """Create a criterion config with a criterion that has registered buffers."""
    return {
        'class': PyTorchCriterionWrapper,
        'args': {
            'criterion': dummy_criterion,
            'use_buffer': False,
        },
    }


@pytest.fixture
def criterion(criterion_cfg):
    """Create an AuxiliaryOutputsCriterion instance for testing."""
    return AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)


def test_initialization(criterion):
    """Test that the criterion is properly registered as a submodule."""
    # Test that the criterion is properly registered as a submodule
    assert hasattr(criterion, 'criterion')
    assert isinstance(criterion.criterion, PyTorchCriterionWrapper)

    # Test that the criterion is in the module's children
    assert 'criterion' in dict(criterion.named_children())


def test_call_with_list_input(criterion, sample_tensors, sample_tensor):
    """Test calling the criterion with a list of predictions."""
    # Compute loss
    loss = criterion(y_pred=sample_tensors, y_true=sample_tensor)
    criterion._buffer_queue.join()

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


def test_buffer_behavior(criterion_cfg, sample_tensors, sample_tensor):
    """Test the buffer behavior of AuxiliaryOutputsCriterion."""
    # Create a criterion
    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Test initialize
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and criterion.buffer == []
    assert criterion.criterion.use_buffer is False
    assert not hasattr(criterion.criterion, 'buffer')

    # Test update
    loss = criterion(y_pred=sample_tensors, y_true=sample_tensor)
    criterion._buffer_queue.join()
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and len(criterion.buffer) == 1
    assert criterion.buffer[0].equal(loss.detach().cpu())
    assert criterion.criterion.use_buffer is False
    assert not hasattr(criterion.criterion, 'buffer')

    # Test reset
    criterion.reset_buffer()
    assert criterion.use_buffer is True
    assert hasattr(criterion, 'buffer') and criterion.buffer == []
    assert criterion.criterion.use_buffer is False
    assert not hasattr(criterion.criterion, 'buffer')


def test_device_transfer(criterion_cfg, sample_tensors, sample_tensor):
    """Test moving the criterion between CPU and GPU."""
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create a criterion
    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Step 1: Test on CPU
    # Check initial state
    assert not criterion.criterion.criterion.class_weights.is_cuda
    assert len(criterion.buffer) == 0

    # Compute loss on CPU
    cpu_loss = criterion(y_pred=sample_tensors, y_true=sample_tensor)
    criterion._buffer_queue.join()
    assert len(criterion.buffer) == 1

    # Step 2: Move to GPU
    criterion = criterion.cuda()
    gpu_tensors = [t.cuda() for t in sample_tensors]
    gpu_target = sample_tensor.cuda()

    # Check GPU state
    assert criterion.criterion.criterion.class_weights.is_cuda
    assert len(criterion.buffer) == 1

    # Compute loss on GPU
    gpu_loss = criterion(y_pred=gpu_tensors, y_true=gpu_target)
    criterion._buffer_queue.join()
    assert len(criterion.buffer) == 2

    # Step 3: Move back to CPU
    criterion = criterion.cpu()

    # Check CPU state
    assert not criterion.criterion.criterion.class_weights.is_cuda
    assert len(criterion.buffer) == 2

    # Compute loss on CPU again
    cpu_loss2 = criterion(y_pred=sample_tensors, y_true=sample_tensor)
    criterion._buffer_queue.join()
    assert len(criterion.buffer) == 3

    # Check that all losses are equivalent
    assert abs(cpu_loss.item() - gpu_loss.item()) < 1e-5
    assert abs(cpu_loss.item() - cpu_loss2.item()) < 1e-5
