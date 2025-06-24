import pytest
import torch
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.auxiliary_outputs_criterion import AuxiliaryOutputsCriterion


def test_basic_initialization(aux_criterion):
    """Test that the criterion is properly registered as a submodule."""
    # Test that the criterion is properly registered as a submodule
    assert hasattr(aux_criterion, 'criterion')
    assert isinstance(aux_criterion.criterion, PyTorchCriterionWrapper)

    # Test that the criterion is in the module's children
    assert 'criterion' in dict(aux_criterion.named_children())


def test_inheritance_verification(dummy_criterion):
    """Test that AuxiliaryOutputsCriterion properly inherits from SingleTaskCriterion."""
    from criteria.wrappers.single_task_criterion import SingleTaskCriterion
    from criteria.base_criterion import BaseCriterion

    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Test inheritance chain
    assert isinstance(criterion, SingleTaskCriterion)
    assert isinstance(criterion, BaseCriterion)
    assert isinstance(criterion, torch.nn.Module)

    # Test that it has inherited attributes and methods
    assert hasattr(criterion, 'use_buffer')
    assert hasattr(criterion, 'add_to_buffer')
    assert hasattr(criterion, 'reset_buffer')
    assert hasattr(criterion, 'summarize')


def test_reduction_options_validation(dummy_criterion):
    """Test that reduction options are properly validated."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    # Test valid options
    for valid_option in AuxiliaryOutputsCriterion.REDUCTION_OPTIONS:
        criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg, reduction=valid_option)
        assert criterion.reduction == valid_option

    # Test invalid options
    with pytest.raises(AssertionError):
        AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg, reduction='invalid')

    with pytest.raises(AssertionError):
        AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg, reduction='multiply')


def test_component_criterion_buffer_assertion(dummy_criterion):
    """Test that component criterion with buffer raises assertion error."""
    criterion_cfg_with_buffer = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': True}  # This should cause error
    }

    with pytest.raises(AssertionError, match="should not use buffer"):
        AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg_with_buffer)


def test_default_buffer_enabled(aux_criterion):
    """Test that buffer is enabled by default."""
    assert aux_criterion.use_buffer is True
    assert hasattr(aux_criterion, 'buffer')
    assert isinstance(aux_criterion.buffer, list)
    assert len(aux_criterion.buffer) == 0


def test_component_criterion_buffer_disabled(aux_criterion):
    """Test that component criterion has buffer disabled."""
    assert aux_criterion.criterion.use_buffer is False
    assert not hasattr(aux_criterion.criterion, 'buffer')


def test_default_reduction_option(dummy_criterion):
    """Test that default reduction option is 'sum'."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)
    assert criterion.reduction == 'sum'
