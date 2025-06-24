import pytest
import torch
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.hybrid_criterion import HybridCriterion
from utils.builders import build_from_config


def test_build_from_config_integration(dummy_criterion):
    """Test that HybridCriterion works correctly with build_from_config."""
    hybrid_config = {
        'class': HybridCriterion,
        'args': {
            'combine': 'sum',
            'criteria_cfg': [
                {
                    'class': PyTorchCriterionWrapper,
                    'args': {'criterion': dummy_criterion}
                },
                {
                    'class': PyTorchCriterionWrapper,
                    'args': {'criterion': dummy_criterion}
                }
            ]
        }
    }
    
    # Build criterion from config
    criterion = build_from_config(hybrid_config)
    
    # Verify it was built correctly
    assert isinstance(criterion, HybridCriterion)
    assert len(criterion.criteria) == 2
    assert criterion.use_buffer is True
    assert criterion.combine == 'sum'
    
    # Test that it works functionally
    sample_input = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    sample_target = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    
    loss = criterion(y_pred=sample_input, y_true=sample_target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_build_from_config_with_buffer_disabled(dummy_criterion):
    """Test building HybridCriterion with disabled buffer from config."""
    hybrid_config = {
        'class': HybridCriterion,
        'args': {
            'use_buffer': False,
            'combine': 'mean',
            'criteria_cfg': [
                {
                    'class': PyTorchCriterionWrapper,
                    'args': {'criterion': dummy_criterion}
                }
            ]
        }
    }
    
    criterion = build_from_config(hybrid_config)
    
    assert isinstance(criterion, HybridCriterion)
    assert criterion.use_buffer is False
    assert not hasattr(criterion, 'buffer')
    assert criterion.combine == 'mean'


def test_nested_config_building(dummy_criterion):
    """Test that nested configs are built correctly."""
    # Create a more complex nested config
    complex_config = {
        'class': HybridCriterion,
        'args': {
            'combine': 'mean',
            'criteria_cfg': [
                {
                    'class': PyTorchCriterionWrapper,
                    'args': {
                        'criterion': dummy_criterion,
                        'use_buffer': True  # This should be overridden
                    }
                },
                {
                    'class': PyTorchCriterionWrapper,
                    'args': {
                        'criterion': dummy_criterion,
                        'use_buffer': True  # This should be overridden
                    }
                }
            ]
        }
    }
    
    criterion = build_from_config(complex_config)
    
    # Verify nested components were built correctly with overridden buffer settings
    assert len(criterion.criteria) == 2
    for component in criterion.criteria:
        assert component.use_buffer is False
        assert not hasattr(component, 'buffer')


def test_config_parameter_merging(dummy_criterion):
    """Test that config parameters are properly merged during building."""
    base_config = {
        'class': HybridCriterion,
        'args': {
            'combine': 'sum',
            'criteria_cfg': [
                {
                    'class': PyTorchCriterionWrapper,
                    'args': {'criterion': dummy_criterion}
                }
            ]
        }
    }
    
    # Test building with additional kwargs
    criterion = build_from_config(base_config, use_buffer=False)
    
    assert criterion.use_buffer is False
    assert not hasattr(criterion, 'buffer')


def test_recursive_building_preservation(dummy_criterion):
    """Test that recursive building preserves object references correctly."""
    # Create config with shared references
    shared_criterion_config = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion}
    }
    
    # Use the same config reference in multiple places
    hybrid_config = {
        'class': HybridCriterion,
        'args': {
            'combine': 'sum',
            'criteria_cfg': [shared_criterion_config, shared_criterion_config]
        }
    }
    
    # This should create two separate instances, not share the same instance
    criterion = build_from_config(hybrid_config)
    
    assert len(criterion.criteria) == 2
    # They should be different instances even though built from same config
    assert criterion.criteria[0] is not criterion.criteria[1]
    # But should have same class and attributes
    assert type(criterion.criteria[0]) == type(criterion.criteria[1])


def test_error_handling_in_build_process(dummy_criterion):
    """Test error handling during the build process."""
    # Test with malformed config - missing criterion
    malformed_config = {
        'class': HybridCriterion,
        'args': {
            'combine': 'sum',
            'criteria_cfg': [
                {
                    'class': PyTorchCriterionWrapper,
                    'args': {}  # Missing required criterion
                }
            ]
        }
    }
    
    with pytest.raises(Exception):  # Should fail during component building
        build_from_config(malformed_config)
    
    # Test with completely invalid config structure
    invalid_config = {
        'invalid_key': 'invalid_value'
    }
    
    # This should return the config as-is since it doesn't match expected structure
    result = build_from_config(invalid_config)
    assert result == invalid_config


def test_build_from_config_type_validation():
    """Test that build_from_config validates types correctly."""
    # Test with non-dict config
    non_dict_config = "not_a_dict"
    result = build_from_config(non_dict_config)
    assert result == non_dict_config
    
    # Test with list config
    list_config = [1, 2, 3]
    result = build_from_config(list_config)
    assert result == [1, 2, 3]  # Should process each element
