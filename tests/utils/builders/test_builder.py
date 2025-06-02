import pytest
import torch
import torch.nn as nn
from utils.builders import build_from_config


class DummyClass:
    def __init__(self, value1, value2):
        self.value1 = value1
        self.value2 = value2

    def __eq__(self, other):
        if not isinstance(other, DummyClass):
            return False
        return self.value1 == other.value1 and self.value2 == other.value2


def test_basic_functionality():
    """Test basic functionality of build_from_config with simple objects."""
    # Test with simple values
    assert build_from_config(5) == 5
    assert build_from_config("test") == "test"
    assert build_from_config([1, 2, 3]) == [1, 2, 3]

    # Test with nested config
    config = {
        'class': DummyClass,
        'args': {
            'value1': 1,
            'value2': {
                'class': DummyClass,
                'args': {
                    'value1': 2,
                    'value2': 3
                }
            }
        }
    }
    result = build_from_config(config)
    assert isinstance(result, DummyClass)
    assert result.value1 == 1
    assert isinstance(result.value2, DummyClass)
    assert result.value2.value1 == 2
    assert result.value2.value2 == 3

    # Test with kwargs
    config = {
        'class': DummyClass,
        'args': {
            'value1': 1
        }
    }
    result = build_from_config(config, value2=2)
    assert isinstance(result, DummyClass)
    assert result.value1 == 1
    assert result.value2 == 2


def test_optimizer_parameter_preservation():
    """Test that build_from_config preserves PyTorch parameters when building optimizers."""
    # Create a simple model
    model = nn.Linear(10, 10)
    original_params = list(model.parameters())
    
    # Create optimizer config
    optimizer_config = {
        'class': torch.optim.SGD,
        'args': {
            'params': original_params,
            'lr': 0.01
        }
    }

    # Build optimizer using build_from_config
    optimizer = build_from_config(optimizer_config)
    
    # Get parameters from optimizer
    optimizer_params = optimizer.param_groups[0]['params']

    # Test that parameters are preserved (same objects)
    assert len(original_params) == len(optimizer_params)
    for orig_param, opt_param in zip(original_params, optimizer_params):
        # Test using id()
        assert id(orig_param) == id(opt_param), "Parameters should be the same object (id)"
        # Test using data_ptr()
        assert orig_param.data_ptr() == opt_param.data_ptr(), "Parameters should point to the same memory (data_ptr)"
        # Test using is operator
        assert orig_param is opt_param, "Parameters should be the same object (is)"

    # Test with nested config
    nested_config = {
        'class': DummyClass,
        'args': {
            'value1': optimizer_config,
            'value2': 3
        }
    }
    
    # Build nested config
    nested_result = build_from_config(nested_config)
    nested_optimizer = nested_result.value1
    nested_params = nested_optimizer.param_groups[0]['params']

    # Test that parameters are preserved in nested config
    assert len(original_params) == len(nested_params)
    for orig_param, nested_param in zip(original_params, nested_params):
        assert id(orig_param) == id(nested_param), "Parameters should be preserved in nested config (id)"
        assert orig_param.data_ptr() == nested_param.data_ptr(), "Parameters should be preserved in nested config (data_ptr)"
        assert orig_param is nested_param, "Parameters should be preserved in nested config (is)"

    # Test that modifying optimizer parameters affects model parameters
    for param in optimizer_params:
        param.data += 1.0
    
    for orig_param, opt_param in zip(original_params, optimizer_params):
        assert torch.allclose(orig_param, opt_param), "Parameter modifications should be reflected in both model and optimizer" 