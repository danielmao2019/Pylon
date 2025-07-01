import pytest
import torch
import torch.nn as nn
from typing import Dict, Any, List
from debuggers.base_debugger import BaseDebugger
from debuggers.forward_debugger import ForwardDebugger


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


# Forward Debugger Implementations for Testing
class FeatureMapDebugger(ForwardDebugger):
    """Test debugger for feature map extraction."""
    
    def process_forward(self, module, input, output):
        if isinstance(output, torch.Tensor):
            return {
                'feature_map': output.detach().cpu(),
                'stats': {
                    'mean': float(output.mean()),
                    'std': float(output.std()),
                    'min': float(output.min()),
                    'max': float(output.max()),
                    'shape': list(output.shape)
                },
                'layer_name': self.layer_name
            }
        return {'error': 'Output is not a tensor'}


class ActivationStatsDebugger(ForwardDebugger):
    """Test debugger for activation statistics."""
    
    def process_forward(self, module, input, output):
        if isinstance(output, torch.Tensor):
            activation = output.detach().cpu()
            stats = {
                'shape': list(activation.shape),
                'mean': float(activation.mean()),
                'std': float(activation.std()),
                'min': float(activation.min()),
                'max': float(activation.max()),
                'l1_norm': float(activation.abs().mean()),
                'l2_norm': float(activation.norm()),
                'sparsity': float((activation == 0).float().mean()),
                'positive_ratio': float((activation > 0).float().mean()),
                'saturation_low': float((activation <= -1).float().mean()),
                'saturation_high': float((activation >= 1).float().mean()),
            }
            if len(activation.shape) >= 3:
                channel_means = activation.mean(dim=(0, 2, 3)) if len(activation.shape) == 4 else activation.mean(dim=0)
                stats['channel_means'] = channel_means.tolist()
                stats['channel_stds'] = (activation.std(dim=(0, 2, 3)) if len(activation.shape) == 4 else activation.std(dim=0)).tolist()
            return {
                'layer_name': self.layer_name,
                'module_type': type(module).__name__,
                'activation_stats': stats,
                'sample_values': activation.flatten()[:100].tolist()
            }
        return {'error': 'Output is not a tensor'}


class LayerOutputDebugger(ForwardDebugger):
    """Test debugger for layer outputs."""
    
    def __init__(self, layer_name: str, downsample_factor: int = 4):
        super().__init__(layer_name)
        self.downsample_factor = downsample_factor
        
    def process_forward(self, module, input, output):
        if isinstance(output, torch.Tensor):
            output_cpu = output.detach().cpu()
            if len(output_cpu.shape) >= 3 and self.downsample_factor > 1:
                if len(output_cpu.shape) == 4:
                    downsampled = torch.nn.functional.avg_pool2d(output_cpu, self.downsample_factor)
                elif len(output_cpu.shape) == 3:
                    downsampled = torch.nn.functional.avg_pool2d(output_cpu.unsqueeze(0), self.downsample_factor).squeeze(0)
                else:
                    downsampled = output_cpu
            else:
                downsampled = output_cpu
            return {
                'layer_name': self.layer_name,
                'original_shape': list(output_cpu.shape),
                'output': downsampled,
                'downsampled': self.downsample_factor > 1,
                'downsample_factor': self.downsample_factor
            }
        return {'error': 'Output is not a tensor'}


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