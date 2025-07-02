import pytest
import torch
import torch.nn as nn
from optimizers.single_task_optimizer import SingleTaskOptimizer


@pytest.fixture
def simple_model():
    """Simple model for testing."""
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )


@pytest.fixture
def multi_part_optimizer_configs(simple_model):
    """Standard multi-part optimizer configuration for testing."""
    return {
        'encoder': {
            'class': SingleTaskOptimizer,
            'args': {
                'optimizer_config': {
                    'class': torch.optim.SGD,
                    'args': {
                        'params': list(simple_model[0].parameters()),
                        'lr': 0.01,
                        'momentum': 0.9
                    }
                }
            }
        },
        'decoder': {
            'class': SingleTaskOptimizer,
            'args': {
                'optimizer_config': {
                    'class': torch.optim.SGD,
                    'args': {
                        'params': list(simple_model[2].parameters()),
                        'lr': 0.001,
                        'momentum': 0.95
                    }
                }
            }
        }
    }


@pytest.fixture
def basic_optimizer_configs():
    """Basic optimizer configuration for testing without model dependency."""
    return {
        'part1': {
            'class': SingleTaskOptimizer,
            'args': {
                'optimizer_config': {
                    'class': torch.optim.SGD,
                    'args': {
                        'params': [torch.nn.Parameter(torch.randn(2, 2))],
                        'lr': 0.01
                    }
                }
            }
        },
        'part2': {
            'class': SingleTaskOptimizer,
            'args': {
                'optimizer_config': {
                    'class': torch.optim.SGD,
                    'args': {
                        'params': [torch.nn.Parameter(torch.randn(2, 2))],
                        'lr': 0.01
                    }
                }
            }
        }
    }
