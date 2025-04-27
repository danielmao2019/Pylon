import pytest
import torch
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.multi_task_criterion import MultiTaskCriterion


@pytest.fixture
def criterion_cfgs(dummy_criterion):
    """Create criterion configs with criteria that have registered buffers."""
    return {
        'task1': {
            'class': PyTorchCriterionWrapper,
            'args': {
                'criterion': dummy_criterion
            }
        },
        'task2': {
            'class': PyTorchCriterionWrapper,
            'args': {
                'criterion': dummy_criterion
            }
        }
    }


@pytest.fixture
def criterion(criterion_cfgs):
    """Create a MultiTaskCriterion instance for testing."""
    return MultiTaskCriterion(criterion_cfgs=criterion_cfgs)


def test_initialization(criterion):
    """Test that the criteria are properly registered as submodules."""
    # Test that the criteria are properly registered as submodules
    assert hasattr(criterion, 'task_criteria')
    assert isinstance(criterion.task_criteria, torch.nn.ModuleDict)
    assert len(criterion.task_criteria) == 2

    # Test that the criteria are in the module's children
    children = dict(criterion.named_children())
    assert 'task_criteria' in children

    # Test that each criterion is properly registered
    assert isinstance(criterion.task_criteria['task1'], PyTorchCriterionWrapper)
    assert isinstance(criterion.task_criteria['task2'], PyTorchCriterionWrapper)

    # Test that task_names is set correctly
    assert criterion.task_names == {'task1', 'task2'}


def test_reset_buffer(criterion, sample_multi_task_tensors):
    """Test resetting the buffer."""
    assert not hasattr(criterion, 'buffer')
    assert hasattr(criterion, 'task_criteria')
    assert isinstance(criterion.task_criteria, torch.nn.ModuleDict)
    assert all(hasattr(task_criterion, 'buffer') for task_criterion in criterion.task_criteria.values())

    # Check that each task criterion's buffer has been reset
    for task_criterion in criterion.task_criteria.values():
        assert len(task_criterion.buffer) == 0

    # Call the criterion to add to buffer
    criterion(y_pred=sample_multi_task_tensors, y_true=sample_multi_task_tensors)

    # Check that each task criterion's buffer has been reset
    for task_criterion in criterion.task_criteria.values():
        assert len(task_criterion.buffer) == 1

    # Reset the buffer
    criterion.reset_buffer()

    # Check that each task criterion's buffer has been reset
    for task_criterion in criterion.task_criteria.values():
        assert len(task_criterion.buffer) == 0


def test_call(criterion, sample_multi_task_tensors):
    """Test calling the criterion."""
    # Call the criterion
    losses = criterion(y_pred=sample_multi_task_tensors, y_true=sample_multi_task_tensors)

    # Check that losses is a dictionary with the correct keys
    assert isinstance(losses, dict)
    assert set(losses.keys()) == {'task1', 'task2'}

    # Check that each loss is a scalar tensor
    for loss in losses.values():
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    # Check that each task criterion's buffer has been updated
    for task_criterion in criterion.task_criteria.values():
        assert len(task_criterion.buffer) == 1


def test_summarize(criterion, sample_multi_task_tensors, tmp_path):
    """Test summarizing the criterion."""
    # Call the criterion to add to buffer
    criterion(y_pred=sample_multi_task_tensors, y_true=sample_multi_task_tensors)

    # Summarize the criterion
    output_path = tmp_path / "summary.pt"
    result = criterion.summarize(output_path=str(output_path))

    # Check that result is a dictionary with the correct keys
    assert isinstance(result, dict)
    assert set(result.keys()) == {'task1', 'task2'}

    # Check that each result is a tensor
    for tensor in result.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 1

    # Check that the output file exists
    assert output_path.exists()


def test_device_transfer(criterion_cfgs, sample_multi_task_tensors):
    """Test moving the criterion between CPU and GPU."""
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create a criterion
    criterion = MultiTaskCriterion(criterion_cfgs=criterion_cfgs)
    
    # Step 1: Test on CPU
    # Check initial state
    assert not next(criterion.parameters()).is_cuda
    for task_criterion in criterion.task_criteria.values():
        assert not next(task_criterion.parameters()).is_cuda
        assert not task_criterion.criterion.class_weights.is_cuda
    assert len(criterion.buffer) == 0
    
    # Compute loss on CPU
    cpu_losses = criterion(y_pred=sample_multi_task_tensors, y_true=sample_multi_task_tensors)
    assert len(criterion.buffer) == 1
    
    # Step 2: Move to GPU
    criterion = criterion.cuda()
    gpu_tensors = {k: v.cuda() for k, v in sample_multi_task_tensors.items()}
    
    # Check GPU state
    assert next(criterion.parameters()).is_cuda
    for task_criterion in criterion.task_criteria.values():
        assert next(task_criterion.parameters()).is_cuda
        assert task_criterion.criterion.class_weights.is_cuda
    assert len(criterion.buffer) == 1
    
    # Compute loss on GPU
    gpu_losses = criterion(y_pred=gpu_tensors, y_true=gpu_tensors)
    assert len(criterion.buffer) == 2
    
    # Step 3: Move back to CPU
    criterion = criterion.cpu()
    
    # Check CPU state
    assert not next(criterion.parameters()).is_cuda
    for task_criterion in criterion.task_criteria.values():
        assert not next(task_criterion.parameters()).is_cuda
        assert not task_criterion.criterion.class_weights.is_cuda
    assert len(criterion.buffer) == 2
    
    # Compute loss on CPU again
    cpu_losses2 = criterion(y_pred=sample_multi_task_tensors, y_true=sample_multi_task_tensors)
    assert len(criterion.buffer) == 3
    
    # Check that all losses are equivalent
    for task_name in criterion.task_names:
        assert abs(cpu_losses[task_name].item() - gpu_losses[task_name].item()) < 1e-5
        assert abs(cpu_losses[task_name].item() - cpu_losses2[task_name].item()) < 1e-5
