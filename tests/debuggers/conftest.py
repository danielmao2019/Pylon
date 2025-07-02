import pytest
import torch
import torch.nn as nn
from typing import Dict, Any, List
from debuggers.base_debugger import BaseDebugger


class DummyModel(nn.Module):
    """A simple model for testing debugger functionality."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DummyDebugger(BaseDebugger):
    """A dummy debugger that captures simple statistics."""

    def __init__(self, output_key: str = "dummy_stats"):
        self.output_key = output_key

    def __call__(self, datapoint: Dict[str, Dict[str, Any]], model: torch.nn.Module) -> Dict[str, Any]:
        outputs = datapoint['outputs']
        stats = {
            'mean': torch.mean(outputs).item(),
            'std': torch.std(outputs).item(),
            'min': torch.min(outputs).item(),
            'max': torch.max(outputs).item(),
        }
        return {self.output_key: stats}


class AnotherDummyDebugger(BaseDebugger):
    """Another dummy debugger for testing combinations."""

    def __init__(self, output_key: str = "another_stats"):
        self.output_key = output_key

    def __call__(self, datapoint: Dict[str, Dict[str, Any]], model: torch.nn.Module) -> Dict[str, Any]:
        inputs = datapoint['inputs']
        # Simple input analysis
        stats = {
            'input_shape': list(inputs.shape),
            'input_mean': torch.mean(inputs).item(),
        }
        return {self.output_key: stats}


# NOTE: Forward debugger test classes are now defined directly in test_forward_debugger.py
# conftest.py is for fixtures only, not class definitions


@pytest.fixture
def dummy_model():
    """Fixture that provides a DummyModel instance."""
    return DummyModel()


@pytest.fixture
def sample_datapoint():
    """Create a sample datapoint for testing."""
    batch_size = 2
    return {
        'inputs': torch.randn(batch_size, 3, 32, 32, dtype=torch.float32),
        'labels': torch.randint(0, 10, (batch_size,), dtype=torch.int64),
        'outputs': torch.randn(batch_size, 10, dtype=torch.float32),
        'meta_info': {
            'idx': [0],  # List format as used in BaseTrainer/BaseEvaluator
            'image_path': ['/path/to/image.jpg'],
        }
    }


@pytest.fixture
def sample_datapoint_tensor_idx():
    """Create a sample datapoint with tensor idx format."""
    batch_size = 2
    return {
        'inputs': torch.randn(batch_size, 3, 32, 32, dtype=torch.float32),
        'labels': torch.randint(0, 10, (batch_size,), dtype=torch.int64),
        'outputs': torch.randn(batch_size, 10, dtype=torch.float32),
        'meta_info': {
            'idx': torch.tensor([0], dtype=torch.int64),  # Tensor format
            'image_path': ['/path/to/image.jpg'],
        }
    }


@pytest.fixture
def sample_datapoint_int_idx():
    """Create a sample datapoint with direct int idx format."""
    batch_size = 2
    return {
        'inputs': torch.randn(batch_size, 3, 32, 32, dtype=torch.float32),
        'labels': torch.randint(0, 10, (batch_size,), dtype=torch.int64),
        'outputs': torch.randn(batch_size, 10, dtype=torch.float32),
        'meta_info': {
            'idx': 0,  # Direct int format
            'image_path': ['/path/to/image.jpg'],
        }
    }


@pytest.fixture
def dummy_debugger():
    """Fixture that provides a DummyDebugger instance."""
    return DummyDebugger(output_key="test_stats")


@pytest.fixture
def another_dummy_debugger():
    """Fixture that provides an AnotherDummyDebugger instance."""
    return AnotherDummyDebugger(output_key="input_analysis")


@pytest.fixture
def debuggers_config():
    """Create debugger configs for testing SequentialDebugger."""
    return [
        {
            'name': 'dummy_stats',
            'debugger_config': {
                'class': DummyDebugger,
                'args': {
                    'output_key': 'dummy_stats'
                }
            }
        },
        {
            'name': 'input_analysis',
            'debugger_config': {
                'class': AnotherDummyDebugger,
                'args': {
                    'output_key': 'input_analysis'
                }
            }
        }
    ]


@pytest.fixture
def forward_debugger_config():
    """Create forward debugger config for testing."""
    return {
        'name': 'conv2_features',
        'debugger_config': {
            'class': FeatureMapDebugger,
            'args': {
                'layer_name': 'conv2'
            }
        }
    }


@pytest.fixture
def mixed_debuggers_config(forward_debugger_config):
    """Create mixed debugger configs (forward + regular) for testing."""
    return [
        {
            'name': 'dummy_stats',
            'debugger_config': {
                'class': DummyDebugger,
                'args': {
                    'output_key': 'dummy_stats'
                }
            }
        },
        forward_debugger_config
    ]


@pytest.fixture
def empty_debuggers_config():
    """Create empty debugger config for edge case testing."""
    return []