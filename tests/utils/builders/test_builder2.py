import pytest
import torch
import torch.nn as nn
from utils.builders.builder import build_from_config


class DummyClass:
    def __init__(self, value1, value2):
        self.value1 = value1
        self.value2 = value2

    def __eq__(self, other):
        if not isinstance(other, DummyClass):
            return False
        return self.value1 == other.value1 and self.value2 == other.value2


@pytest.mark.parametrize(
    "input_config, expected",
    [
        (5, 5),
        ("test", "test"),
        ([1, 2, 3], [1, 2, 3]),
        (
            {
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
            },
            DummyClass(1, DummyClass(2, 3))
        ),
        (
            {
                'class': DummyClass,
                'args': {
                    'value1': 1
                }
            },
            DummyClass(1, 2)  # We'll pass value2=2 as a kwarg in the test
        ),
    ]
)
def test_basic_functionality(input_config, expected):
    if isinstance(expected, DummyClass) and 'value2' not in input_config.get('args', {}):
        # Test with kwargs for DummyClass
        result = build_from_config(input_config, value2=2)
        assert result == expected
    else:
        result = build_from_config(input_config)
        assert result == expected


@pytest.mark.parametrize(
    "optimizer_config_factory",
    [
        # Direct optimizer config
        lambda params: {
            'class': torch.optim.SGD,
            'args': {
                'params': params,
                'lr': 0.01
            }
        },
        # Nested optimizer config inside DummyClass
        lambda params: {
            'class': DummyClass,
            'args': {
                'value1': {
                    'class': torch.optim.SGD,
                    'args': {
                        'params': params,
                        'lr': 0.01
                    }
                },
                'value2': 3
            }
        },
    ]
)
def test_optimizer_parameter_preservation(optimizer_config_factory):
    model = nn.Linear(10, 10)
    original_params = list(model.parameters())
    config = optimizer_config_factory(original_params)
    result = build_from_config(config)

    # Extract optimizer depending on config type
    if isinstance(result, torch.optim.Optimizer):
        optimizer = result
    elif isinstance(result, DummyClass) and isinstance(result.value1, torch.optim.Optimizer):
        optimizer = result.value1
    else:
        raise AssertionError("Unexpected result type from build_from_config")

    optimizer_params = optimizer.param_groups[0]['params']
    assert len(original_params) == len(optimizer_params)
    for orig_param, opt_param in zip(original_params, optimizer_params):
        assert id(orig_param) == id(opt_param), "Parameters should be the same object (id)"
        assert orig_param.data_ptr() == opt_param.data_ptr(), "Parameters should point to the same memory (data_ptr)"
        assert orig_param is opt_param, "Parameters should be the same object (is)"

    # Test that modifying optimizer parameters affects model parameters
    for param in optimizer_params:
        param.data += 1.0
    for orig_param, opt_param in zip(original_params, optimizer_params):
        assert torch.allclose(orig_param, opt_param), "Parameter modifications should be reflected in both model and optimizer"
