import pytest
import torch
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.hybrid_criterion import HybridCriterion


def test_basic_initialization(hybrid_criterion):
    """Test that the criteria are properly registered as submodules."""
    # Test that the criteria are properly registered as submodules
    assert hasattr(hybrid_criterion, 'criteria')
    assert isinstance(hybrid_criterion.criteria, torch.nn.ModuleList)
    assert len(hybrid_criterion.criteria) == 2

    # Test that the criteria are in the module's children
    children = dict(hybrid_criterion.named_children())
    assert 'criteria' in children

    # Test that each criterion is properly registered
    assert isinstance(hybrid_criterion.criteria[0], PyTorchCriterionWrapper)
    assert isinstance(hybrid_criterion.criteria[1], PyTorchCriterionWrapper)


def test_inheritance_verification(dummy_criterion):
    """Test that HybridCriterion properly inherits from SingleTaskCriterion."""
    from criteria.wrappers.single_task_criterion import SingleTaskCriterion
    from criteria.base_criterion import BaseCriterion
    
    criteria_cfg = [
        {
            'class': PyTorchCriterionWrapper,
            'args': {'criterion': dummy_criterion}
        }
    ]
    
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    
    # Test inheritance chain
    assert isinstance(criterion, SingleTaskCriterion)
    assert isinstance(criterion, BaseCriterion)
    assert isinstance(criterion, torch.nn.Module)
    
    # Test that it has inherited attributes and methods
    assert hasattr(criterion, 'use_buffer')
    assert hasattr(criterion, 'add_to_buffer')
    assert hasattr(criterion, 'reset_buffer')
    assert hasattr(criterion, 'summarize')


def test_combine_options_validation(dummy_criterion):
    """Test that combine options are properly validated."""
    criteria_cfg = [
        {
            'class': PyTorchCriterionWrapper,
            'args': {'criterion': dummy_criterion}
        }
    ]
    
    # Test valid options
    for valid_option in HybridCriterion.COMBINE_OPTIONS:
        criterion = HybridCriterion(combine=valid_option, criteria_cfg=criteria_cfg)
        assert criterion.combine == valid_option
    
    # Test invalid options
    with pytest.raises(AssertionError):
        HybridCriterion(combine='invalid', criteria_cfg=criteria_cfg)
    
    with pytest.raises(AssertionError):
        HybridCriterion(combine='multiply', criteria_cfg=criteria_cfg)


def test_empty_criteria_config():
    """Test that empty criteria config raises assertion error."""
    with pytest.raises(AssertionError):
        HybridCriterion(combine='sum', criteria_cfg=[])
    
    with pytest.raises(AssertionError):
        HybridCriterion(combine='sum', criteria_cfg=None)


def test_default_buffer_enabled(hybrid_criterion):
    """Test that buffer is enabled by default."""
    assert hybrid_criterion.use_buffer is True
    assert hasattr(hybrid_criterion, 'buffer')
    assert isinstance(hybrid_criterion.buffer, list)
    assert len(hybrid_criterion.buffer) == 0


def test_component_criteria_buffer_disabled(hybrid_criterion):
    """Test that component criteria have buffers disabled."""
    for component_criterion in hybrid_criterion.criteria:
        assert component_criterion.use_buffer is False
        assert not hasattr(component_criterion, 'buffer')
