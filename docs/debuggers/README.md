# Pylon Debuggers Module

This module provides a framework for capturing and visualizing debugging outputs during model training, validation, and evaluation. It follows Pylon's design patterns and integrates seamlessly with both BaseTrainer and BaseEvaluator pipelines.

## Overview

The debuggers system enables researchers to:
- Capture intermediate model outputs (feature maps, attention maps, etc.)
- Save debugging data during training validation (at checkpoint epochs) or evaluation (always)
- Visualize debugging outputs in the eval viewer alongside input data
- Identify patterns when algorithms fail

## Architecture

### Core Components

1. **BaseDebugger**: Abstract base class for all debuggers
2. **ForwardDebugger**: Base class for debuggers using PyTorch forward hooks
3. **SequentialDebugger**: Wrapper that manages multiple debuggers and handles async saving
4. **Utilities**: Helper functions for layer access and data processing

### Key Features

- **Thread-safe async saving**: Uses background threads and queues (following base_criterion.py pattern)
- **Page-based storage**: Memory-efficient saving with configurable page sizes (dict format)
- **Selective saving**: Only saves at checkpoint epochs to manage storage
- **Flexible debugger types**: Support for both forward hooks and post-forward analysis

## Usage

### 1. Define Custom Debuggers

```python
from debuggers.forward_debugger import ForwardDebugger
from debuggers.base_debugger import BaseDebugger

class MyFeatureDebugger(ForwardDebugger):
    def process_forward(self, module, input, output):
        return {
            'feature_map': output.detach().cpu(),
            'stats': {
                'mean': float(output.mean()),
                'std': float(output.std())
            }
        }

class MyGradCAMDebugger(BaseDebugger):
    def __init__(self, target_layer: str, target_class: int = None):
        self.target_layer = target_layer
        self.target_class = target_class
        
    def __call__(self, datapoint, model):
        # Compute GradCAM using datapoint, model and target layer
        # This is a post-forward debugger that operates on the full datapoint
        inputs = datapoint['inputs']
        outputs = datapoint['outputs']
        # Now we have access to the model for gradient computation
        # ... actual GradCAM computation logic using model ...
        return {'gradcam': computed_gradcam}
```

### 2. Configure in Training Config

```python
from debuggers.wrappers.sequential_debugger import SequentialDebugger

config = {
    'checkpoint_method': 5,  # Save every 5 epochs
    'debugger': {
        'class': SequentialDebugger,
        'args': {
            'page_size_mb': 100,
            'debuggers_config': [
                {
                    'name': 'backbone_features',
                    'debugger_config': {
                        'class': MyFeatureDebugger,
                        'args': {'layer_name': 'backbone.layer4'}
                    }
                },
                {
                    'name': 'gradcam_analysis',
                    'debugger_config': {
                        'class': MyGradCAMDebugger,
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

### 3. Run Training or Evaluation

For training:
```bash
python main.py --config-filepath configs/my_config_with_debugger.py
```

For evaluation:
```bash
python eval.py --config-filepath configs/my_evaluator_config_with_debugger.py
```

### 4. View Results

Debug outputs automatically appear in the eval viewer as a fourth section when viewing datapoints:
- **BaseTrainer**: From checkpoint epochs (e.g., every 5 epochs)
- **BaseEvaluator**: From all evaluated datapoints

## File Structure

**BaseTrainer** (training with debugging enabled):
```
work_dir/
├── epoch_4/
│   ├── checkpoint.pt
│   ├── validation_scores.json
│   └── debugger/
│       ├── page_0.pkl
│       └── page_1.pkl
├── epoch_9/
│   ├── checkpoint.pt
│   ├── validation_scores.json
│   └── debugger/
│       └── page_0.pkl
```

**BaseEvaluator** (evaluation with debugging enabled):
```
work_dir/
├── evaluation_scores.json
└── debugger/
    ├── page_0.pkl
    ├── page_1.pkl
    └── page_2.pkl
```

## Example Debuggers

See [examples.md](examples.md) for comprehensive implementation examples:

- **FeatureMapDebugger**: Captures feature maps with statistics
- **AttentionDebugger**: Extracts attention patterns  
- **GradCAMDebugger**: Computes GradCAM visualizations
- **ActivationStatsDebugger**: Collects detailed activation statistics
- **LayerOutputDebugger**: Saves raw layer outputs with downsampling

## Integration Points

### BaseTrainer Integration

- `_init_debugger()`: Builds debugger with model parameter (forward hooks registered automatically)
- `_init_checkpoint_indices()`: Precomputes when to save debug outputs
- `_before_val_loop()`: Enables/disables debugger based on checkpoint epochs
- `_eval_step()`: Calls debugger with datapoint parameter (buffering handled internally)
- `_after_val_loop_()`: Saves debug outputs to disk

### BaseEvaluator Integration

- `_init_debugger()`: Builds debugger with model parameter (forward hooks registered automatically)
- `_eval_epoch_()`: Enables debugger for all evaluation datapoints
- `_eval_step()`: Calls debugger with datapoint parameter (buffering handled internally)
- `_after_eval_loop_()`: Saves debug outputs to disk

### Eval Viewer Integration

- `load_debug_outputs()`: Loads debug data from page files
- `display_debug_outputs()`: Renders debug visualizations
- Updated display functions support debug sections
- Datapoint viewer automatically includes debug data when available

## Memory Management

- **Page-based storage**: Configurable page size limits (default 100MB)
- **Async processing**: Background threads prevent blocking training
- **Deep size calculation**: Recursive memory usage tracking
- **Thread-safe operations**: Proper locking and queue management

## API Reference

### BaseDebugger

Abstract base class for all debuggers. Must implement:

```python
def __call__(self, datapoint: Dict[str, Dict[str, Any]], model: torch.nn.Module) -> Dict[str, Any]:
    """Process datapoint and return debug outputs."""
    pass
```

### ForwardDebugger

Base class for debuggers using PyTorch forward hooks. Must implement:

```python
def process_forward(self, module: torch.nn.Module, input: Any, output: Any) -> Any:
    """Process data from forward hook."""
    pass
```

### SequentialDebugger

Wrapper that manages multiple debuggers:

```python
def __init__(self, debuggers_config: List[Dict[str, Any]], model: torch.nn.Module, page_size_mb: int = 100):
    """Initialize with debugger configs and model for hook registration."""
    pass

def __call__(self, datapoint: Dict[str, Dict[str, Any]], model: torch.nn.Module) -> Dict[str, Any]:
    """Run all debuggers and handle buffering internally."""
    pass
```

## Design Principles

1. **Flexibility**: Users define custom debuggers for specific needs
2. **Integration**: Follows existing Pylon patterns (criterion, metric, optimizer)
3. **Efficiency**: Memory-efficient with async saving
4. **Simplicity**: Minimal changes to existing codebase
5. **Extensibility**: Easy to add new debugger types

## Testing

The module includes comprehensive tests following Pylon's testing patterns:

```bash
# Run all debugger tests
pytest tests/debuggers/ -v

# Run specific test modules
pytest tests/debuggers/test_base_debugger.py -v
pytest tests/debuggers/test_forward_debugger.py -v
pytest tests/debuggers/wrappers/sequential_debugger/ -v
```

Tests cover:
- Initialization and configuration
- Forward hook functionality  
- Memory management
- Error handling
- API integration
- Threading setup
- Edge cases