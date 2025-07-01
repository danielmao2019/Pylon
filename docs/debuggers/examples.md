# Debugger Examples

This document provides comprehensive examples of how to implement different types of debuggers for the Pylon debuggers module.

## ForwardDebugger Examples

These debuggers use PyTorch forward hooks to capture intermediate outputs during model forward passes.

### 1. FeatureMapDebugger

Captures feature maps with basic statistics:

```python
from typing import Any, Dict
import torch
from debuggers.forward_debugger import ForwardDebugger

class FeatureMapDebugger(ForwardDebugger):
    """Example debugger for extracting feature maps from forward hooks."""

    def process_forward(self, module: torch.nn.Module, input: Any, output: Any) -> Dict[str, Any]:
        """Extract and process feature maps."""
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
```

### 2. AttentionDebugger

Extracts attention patterns and computes attention statistics:

```python
from typing import Any, Dict
import torch
from debuggers.forward_debugger import ForwardDebugger

class AttentionDebugger(ForwardDebugger):
    """Example debugger for extracting attention maps."""

    def process_forward(self, module: torch.nn.Module, input: Any, output: Any) -> Dict[str, Any]:
        """Extract attention patterns."""
        if isinstance(output, torch.Tensor):
            # Assume output is attention weights
            attention = output.detach().cpu()

            # Get attention statistics
            attention_stats = {
                'entropy': float(-torch.sum(attention * torch.log(attention + 1e-8), dim=-1).mean()),
                'max_attention': float(attention.max()),
                'attention_distribution': attention.mean(dim=0) if len(attention.shape) > 1 else attention
            }

            return {
                'attention_map': attention,
                'attention_stats': attention_stats,
                'layer_name': self.layer_name
            }
        return {'error': 'Output is not a tensor'}
```

### 3. ActivationStatsDebugger

Collects comprehensive activation statistics:

```python
from typing import Any, Dict
import torch
from debuggers.forward_debugger import ForwardDebugger

class ActivationStatsDebugger(ForwardDebugger):
    """Example debugger for collecting activation statistics."""

    def process_forward(self, module: torch.nn.Module, input: Any, output: Any) -> Dict[str, Any]:
        """Collect detailed activation statistics."""
        if isinstance(output, torch.Tensor):
            activation = output.detach().cpu()

            # Compute comprehensive statistics
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

            # Add channel-wise statistics if applicable
            if len(activation.shape) >= 3:  # Has channel dimension
                channel_means = activation.mean(dim=(0, 2, 3)) if len(activation.shape) == 4 else activation.mean(dim=0)
                stats['channel_means'] = channel_means.tolist()
                stats['channel_stds'] = (activation.std(dim=(0, 2, 3)) if len(activation.shape) == 4 else activation.std(dim=0)).tolist()

            return {
                'layer_name': self.layer_name,
                'module_type': type(module).__name__,
                'activation_stats': stats,
                'sample_values': activation.flatten()[:100].tolist()  # First 100 values for inspection
            }
        return {'error': 'Output is not a tensor'}
```

### 4. LayerOutputDebugger

Saves raw layer outputs with optional downsampling:

```python
from typing import Any, Dict
import torch
import torch.nn.functional as F
from debuggers.forward_debugger import ForwardDebugger

class LayerOutputDebugger(ForwardDebugger):
    """Simple debugger that saves raw layer outputs."""

    def __init__(self, layer_name: str, downsample_factor: int = 4):
        super().__init__(layer_name)
        self.downsample_factor = downsample_factor

    def process_forward(self, module: torch.nn.Module, input: Any, output: Any) -> Dict[str, Any]:
        """Save layer output with optional downsampling."""
        if isinstance(output, torch.Tensor):
            output_cpu = output.detach().cpu()

            # Downsample if output is too large
            if len(output_cpu.shape) >= 3 and self.downsample_factor > 1:
                if len(output_cpu.shape) == 4:  # (B, C, H, W)
                    downsampled = F.avg_pool2d(output_cpu, self.downsample_factor)
                elif len(output_cpu.shape) == 3:  # (B, H, W) or (C, H, W)
                    downsampled = F.avg_pool2d(output_cpu.unsqueeze(0), self.downsample_factor).squeeze(0)
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
```

## BaseDebugger Examples

These debuggers operate on complete datapoints after the forward pass.

### 5. GradCAMDebugger

Computes GradCAM visualizations:

```python
from typing import Any, Dict
import torch
from debuggers.base_debugger import BaseDebugger

class GradCAMDebugger(BaseDebugger):
    """Example post-forward debugger for computing GradCAM."""

    def __init__(self, target_layer: str, target_class: int = None):
        self.target_layer = target_layer
        self.target_class = target_class

    def __call__(self, datapoint: Dict[str, Dict[str, Any]], model: torch.nn.Module) -> Dict[str, Any]:
        """Compute GradCAM for the target layer."""
        # Extract inputs and outputs
        inputs = datapoint['inputs']
        outputs = datapoint['outputs']

        # Now we have access to the model for gradient computation
        # In practice, you'd use the model to compute gradients and apply GradCAM algorithm
        # For demonstration, we'll create a simple mock GradCAM
        if isinstance(outputs, torch.Tensor):
            # Mock GradCAM as attention over the spatial dimensions
            B, C = outputs.shape[:2]
            if len(outputs.shape) >= 4:  # Has spatial dimensions
                H, W = outputs.shape[2:4]
                gradcam = torch.rand(B, H, W)  # Mock GradCAM
            else:
                gradcam = torch.rand(B, 8, 8)  # Mock spatial GradCAM

            return {
                'gradcam': gradcam.detach().cpu(),
                'target_layer': self.target_layer,
                'target_class': self.target_class,
                'gradcam_stats': {
                    'mean': float(gradcam.mean()),
                    'std': float(gradcam.std()),
                    'shape': list(gradcam.shape)
                }
            }
        else:
            return {'error': 'Model output is not a tensor'}
```

## Usage Examples

### Configuration Example

Here's how to configure these debuggers in your training config:

```python
from debuggers.wrappers.sequential_debugger import SequentialDebugger

config = {
    'checkpoint_method': 5,  # Save debug outputs every 5 epochs
    'debugger': {
        'class': SequentialDebugger,
        'args': {
            'page_size_mb': 100,
            'debuggers_config': [
                {
                    'name': 'backbone_features',
                    'debugger_config': {
                        'class': FeatureMapDebugger,
                        'args': {'layer_name': 'backbone.layer4'}
                    }
                },
                {
                    'name': 'attention_analysis',
                    'debugger_config': {
                        'class': AttentionDebugger,
                        'args': {'layer_name': 'head.attention'}
                    }
                },
                {
                    'name': 'activation_stats',
                    'debugger_config': {
                        'class': ActivationStatsDebugger,
                        'args': {'layer_name': 'backbone.layer3'}
                    }
                },
                {
                    'name': 'raw_outputs',
                    'debugger_config': {
                        'class': LayerOutputDebugger,
                        'args': {
                            'layer_name': 'backbone.layer2',
                            'downsample_factor': 2
                        }
                    }
                },
                {
                    'name': 'gradcam_analysis',
                    'debugger_config': {
                        'class': GradCAMDebugger,
                        'args': {
                            'target_layer': 'backbone.layer4',
                            'target_class': None
                        }
                    }
                }
            ]
        }
    }
}
```

### Custom Implementation Tips

1. **ForwardDebugger Pattern**: Use when you need to capture intermediate outputs during forward pass
   - Implement `process_forward()` method
   - Always call `output.detach().cpu()` to avoid memory issues
   - Handle non-tensor outputs gracefully

2. **BaseDebugger Pattern**: Use when you need access to complete datapoint
   - Implement `__call__()` method taking full datapoint
   - Useful for analysis requiring both inputs and outputs
   - Better for computations like GradCAM, LIME, etc.

3. **Memory Management**: 
   - Tensors are computed on GPU for speed, moved to CPU automatically during saving via apply_tensor_op
   - Use downsampling for large feature maps to reduce memory usage
   - Consider the page_size_mb setting for storage efficiency

4. **Error Handling**:
   - Use explicit type checking (e.g., `isinstance(output, torch.Tensor)`)
   - Return error messages in dict format for graceful failure handling
   - Use assertions for input validation where appropriate

These examples provide a solid foundation for implementing custom debuggers tailored to your specific research needs.
