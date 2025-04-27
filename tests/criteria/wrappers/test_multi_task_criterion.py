import pytest
import torch
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.multi_task_criterion import MultiTaskCriterion


@pytest.fixture
def criterion_cfgs():
    """Create criterion configs for testing."""
    return {
        'task1': {
            'class': PyTorchCriterionWrapper,
            'args': {
                'criterion': torch.nn.MSELoss()
            }
        },
        'task2': {
            'class': PyTorchCriterionWrapper,
            'args': {
                'criterion': torch.nn.L1Loss()
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
