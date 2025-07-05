# Debug Output Visualization Feature Design

## Overview

This feature enables researchers to save and visualize debugging outputs during validation and evaluation to identify patterns when algorithms fail. The system captures intermediate model outputs (feature maps, attention maps, correspondences, etc.) and visualizes them alongside input data in the eval viewer. Works with both BaseTrainer and BaseEvaluator.

## Architecture

### 1. Module Structure

```
debugger/
├── __init__.py
├── base_debugger.py
├── forward_debugger.py
├── utils.py  # Contains get_layer_by_name
└── wrappers/
    ├── __init__.py
    └── sequential_debugger.py
```

### 2. Core Components

#### 2.1 Base Debugger (`debugger/base_debugger.py`)
```python
from typing import Dict, Any
from abc import ABC, abstractmethod
import torch

class BaseDebugger(ABC):
    """Base class for all debuggers."""
    
    @abstractmethod
    def __call__(self, datapoint: Dict[str, Any], model: torch.nn.Module) -> Any:
        """Process datapoint and return debug output.
        
        Args:
            datapoint: Dict with inputs, labels, meta_info, outputs
            model: The model being debugged
            
        Returns:
            Debug output in any format (dict, tensor, list, etc.)
        """
        raise NotImplementedError
```

#### 2.2 Forward Debugger (`debugger/forward_debugger.py`)
```python
from typing import Any, Dict
import torch
from debugger.base_debugger import BaseDebugger

class ForwardDebugger(BaseDebugger):
    """Base class for debuggers that use PyTorch forward hooks."""
    
    def __init__(self, layer_name: str):
        """Initialize forward debugger.
        
        Args:
            layer_name: Dot-separated path to layer (e.g., 'backbone.layer4')
        """
        self.layer_name = layer_name
        self.last_capture = None
        
    def forward_hook_fn(self, module: torch.nn.Module, input: Any, output: Any) -> None:
        """PyTorch forward hook function."""
        self.last_capture = self.process_forward(module, input, output)
        
    @abstractmethod
    def process_forward(self, module: torch.nn.Module, input: Any, output: Any) -> Any:
        """Process data from forward hook.
        
        Args:
            module: The layer this hook is attached to
            input: Input to the layer
            output: Output from the layer
            
        Returns:
            Processed debug data
        """
        raise NotImplementedError
        
    def __call__(self, datapoint: Dict[str, Any], model: torch.nn.Module) -> Any:
        """Return the captured data from forward pass."""
        return self.last_capture
```

#### 2.3 Utilities (`debugger/utils.py`)
```python
from typing import Optional
import torch

def get_layer_by_name(model: torch.nn.Module, layer_name: str) -> Optional[torch.nn.Module]:
    """Get a layer from the model by its name/path.
    
    Args:
        model: The model
        layer_name: Dot-separated path to the layer (e.g., 'backbone.layer4.1.conv2')
        
    Returns:
        The target layer or None if not found
    """
    try:
        layer = model
        for part in layer_name.split('.'):
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                return None
        return layer
    except Exception:
        return None
```

#### 2.4 Sequential Debugger (`debugger/wrappers/sequential_debugger.py`)
```python
from typing import Dict, Any, List, Optional
import os
import threading
import queue
import joblib
import sys
import pickle
from debugger.base_debugger import BaseDebugger
from debugger.forward_debugger import ForwardDebugger
from debugger.utils import get_layer_by_name
from utils.builders import build_from_config

class SequentialDebugger(BaseDebugger):
    """Wrapper that runs multiple debuggers sequentially and manages page-based saving."""
    
    def __init__(self, debuggers_config: List[Dict[str, Any]], 
                 page_size_mb: int = 100):
        """Initialize sequential debugger.
        
        Args:
            debuggers_config: List of debugger configurations
            page_size_mb: Size limit for each page file in MB
        """
        self.page_size = page_size_mb * 1024 * 1024  # Convert to bytes
        self.enabled = False
        
        # Thread and queue management (following base_criterion.py pattern)
        self._buffer_lock = threading.Lock()
        self._buffer_queue = queue.Queue()
        self._buffer_thread = threading.Thread(target=self._buffer_worker, daemon=True)
        self._buffer_thread.start()
        
        # Page management
        self.current_page_idx = 0
        self.current_page_size = 0
        self.current_page_data = {}  # Dict mapping datapoint_idx to debug_outputs
        self.output_dir = None
        
        # Build debuggers from config with name validation
        self.debuggers = {}  # Dict mapping names to debuggers
        self.forward_debuggers = {}  # layer_name -> list of debuggers
        
        debugger_names = set()
        for cfg in debuggers_config:
            name = cfg['name']
            assert name not in debugger_names, f"Duplicate debugger name: {name}"
            debugger_names.add(name)
            
            debugger = build_from_config(cfg['debugger_config'])
            self.debuggers[name] = debugger
            
            # Track forward debuggers by layer for hook registration
            if isinstance(debugger, ForwardDebugger):
                layer_name = debugger.layer_name
                if layer_name not in self.forward_debuggers:
                    self.forward_debuggers[layer_name] = []
                self.forward_debuggers[layer_name].append(debugger)
    
    def __call__(self, datapoint: Dict[str, Any], model: torch.nn.Module, idx: int) -> Dict[str, Any]:
        """Run all debuggers sequentially on the datapoint.
        
        Args:
            datapoint: Dict with inputs, labels, meta_info, outputs
            model: The model being debugged
            idx: Datapoint index for buffer management
            
        Returns:
            Dict mapping debugger names to their outputs
        """
        if not self.enabled:
            return {}
            
        debug_outputs = {}
        for name, debugger in self.debuggers.items():
            debug_outputs[name] = debugger(datapoint, model)
        
        # Handle buffering internally (like criterion and metric do)
        if debug_outputs:  # Only add to buffer if there are debug outputs
            self.add_to_buffer(idx, debug_outputs)
            
        return debug_outputs
    
    def add_to_buffer(self, datapoint_idx: int, debug_outputs: Dict[str, Any]):
        """Add debug outputs to buffer for async processing.
        
        Args:
            datapoint_idx: Index of the datapoint in the dataset
            debug_outputs: Debug outputs from all debuggers
        """
        if not self.enabled:
            return
            
        # Calculate memory size using sys.getsizeof recursively
        data_size = self._get_deep_size(debug_outputs)
        
        # Add to queue for background processing
        self._buffer_queue.put({
            'datapoint_idx': datapoint_idx,
            'debug_outputs': debug_outputs,
            'data_size': data_size
        })
    
    def _buffer_worker(self) -> None:
        """Background thread to handle buffer updates (following base_criterion.py pattern)."""
        while True:
            try:
                item = self._buffer_queue.get()
                datapoint_idx = item['datapoint_idx']
                debug_outputs = item['debug_outputs']
                data_size = item['data_size']
                
                with self._buffer_lock:
                    # Add to current page
                    self.current_page_data[datapoint_idx] = debug_outputs
                    self.current_page_size += data_size
                    
                    # Check if we need to save current page
                    if self.current_page_size >= self.page_size:
                        self._save_current_page()
                
                self._buffer_queue.task_done()
            except Exception as e:
                print(f"Debugger buffer worker error: {e}")
    
    def _get_deep_size(self, obj) -> int:
        """Get the deep memory size of an object using sys.getsizeof recursively."""
        size = sys.getsizeof(obj)
        
        if isinstance(obj, dict):
            size += sum(self._get_deep_size(k) + self._get_deep_size(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set)):
            size += sum(self._get_deep_size(item) for item in obj)
        elif hasattr(obj, '__dict__'):
            size += self._get_deep_size(obj.__dict__)
            
        return size
    
    def _save_current_page(self):
        """Save current page to disk and reset for next page."""
        if not self.current_page_data or not self.output_dir:
            return
            
        page_path = os.path.join(self.output_dir, f"page_{self.current_page_idx}.pkl")
        try:
            joblib.dump(self.current_page_data, page_path)
        except Exception as e:
            print(f"Error saving debug page {page_path}: {e}")
        
        # Reset for next page
        self.current_page_data = []
        self.current_page_size = 0
        self.current_page_idx += 1
    
    def save_all(self, output_dir: str):
        """Save all remaining buffer to disk.
        
        Args:
            output_dir: Directory to save debug outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Wait for all queued items to be processed
        self._buffer_queue.join()
        
        # Save any remaining data in current page
        with self._buffer_lock:
            if self.current_page_data:
                self._save_current_page()
    
    def reset_buffer(self):
        """Reset buffer for new epoch (following base_criterion.py pattern)."""
        # Wait for queue to empty before resetting
        self._buffer_queue.join()
        
        # Assert queue is empty (following base_criterion.py and base_metric.py pattern)
        assert self._buffer_queue.empty(), "Buffer queue is not empty when resetting buffer"
        
        with self._buffer_lock:
            self.current_page_data = {}
            self.current_page_size = 0
            self.current_page_idx = 0
```

### 3. BaseTrainer Integration

#### 3.1 Initialization
```python
# In BaseTrainer._init_components_()
def _init_components_(self):
    self._init_logger()
    self._init_determinism_()
    self._init_state_()
    self._init_dataloaders_()
    self._init_criterion_()
    self._init_metric_()
    self._init_model_()
    self._init_optimizer_()
    self._init_scheduler_()
    self._load_checkpoint_()
    self._init_debugger()  # After model is initialized

def _init_checkpoint_indices(self) -> None:
    """Precompute epoch indices where checkpoints (and debug outputs) will be saved."""
    checkpoint_method = self.config.get('checkpoint_method', 'latest')
    
    if checkpoint_method == 'all':
        self.checkpoint_indices = list(range(self.tot_epochs))
    elif checkpoint_method == 'latest':
        self.checkpoint_indices = [self.tot_epochs - 1]  # Only last epoch
    else:
        # Interval-based: every N epochs
        assert isinstance(checkpoint_method, int) and checkpoint_method > 0
        self.checkpoint_indices = list(range(checkpoint_method-1, self.tot_epochs, checkpoint_method))
        # Always include the last epoch
        if self.tot_epochs - 1 not in self.checkpoint_indices:
            self.checkpoint_indices.append(self.tot_epochs - 1)

def _init_debugger(self):
    """Initialize debugger and register forward hooks."""
    self.logger.info("Initializing debugger...")
    
    # Precompute checkpoint indices
    self._init_checkpoint_indices()
    
    if self.config.get('debugger', None):
        self.debugger = build_from_config(self.config['debugger'])
        
        # Register forward hooks on model
        for layer_name, debuggers in self.debugger.forward_debuggers.items():
            layer = get_layer_by_name(self.model, layer_name)
            if layer is not None:
                for debugger in debuggers:
                    layer.register_forward_hook(debugger.forward_hook_fn)
                self.logger.info(f"Registered {len(debuggers)} forward debugger(s) on layer '{layer_name}'")
            else:
                self.logger.warning(f"Could not find layer '{layer_name}' for debugger")
    else:
        self.debugger = None
```

#### 3.2 Evaluation Integration
```python
# In BaseTrainer._eval_step()
def _eval_step(self, dp: Dict[str, Dict[str, Any]], flush_prefix: Optional[str] = None) -> None:
    # ... existing code through model forward and metric computation ...
    
    # Add debug outputs (only during validation/test at checkpoint indices)
    if self.debugger and self.debugger.enabled:
        dp['debug'] = self.debugger(dp, self.model, idx)
    
    # ... rest of existing code ...

# In BaseTrainer._before_val_loop()
def _before_val_loop(self):
    self.model.eval()
    self.metric.reset_buffer()
    self.logger.eval()
    self.val_dataloader.dataset.set_base_seed(self.val_seeds[self.cum_epochs])
    
    # Enable/disable debugger based on checkpoint indices
    if self.debugger:
        self.debugger.enabled = self.cum_epochs in self.checkpoint_indices
        if self.debugger.enabled:
            self.debugger.reset_buffer()
            self.logger.info(f"Debugger enabled for epoch {self.cum_epochs}")

# In BaseTrainer._after_val_loop_()
def after_val_ops():
    # ... existing saves ...
    
    # Save debugger outputs if enabled
    if self.debugger and self.debugger.enabled:
        debugger_dir = os.path.join(epoch_root, "debugger")
        self.debugger.save_all(debugger_dir)
```

### 4. Eval Viewer Integration

#### 4.1 Loading Debug Outputs
```python
# In runners/eval_viewer/backend/initialization.py
def load_debug_outputs(epoch_dir: str) -> Optional[Dict[int, Any]]:
    """Load all debug outputs for an epoch.
    
    Args:
        epoch_dir: Path to epoch directory
        
    Returns:
        Dict mapping datapoint_idx to debug_outputs, or None if not found
    """
    debugger_dir = os.path.join(epoch_dir, "debugger")
    if not os.path.exists(debugger_dir):
        return None
    
    all_outputs = {}
    # Load all pages in order and merge
    page_files = sorted(glob.glob(os.path.join(debugger_dir, "page_*.pkl")))
    for page_file in page_files:
        page_data = joblib.load(page_file)  # This is now a dict
        all_outputs.update(page_data)  # Merge page dict into all_outputs
    
    return all_outputs
```

#### 4.2 Display Integration
```python
# In data/viewer/layout/display/display_*.py files
def display_datapoint_with_debug(datapoint: Dict[str, Any]) -> html.Div:
    """Display datapoint with debug outputs as fourth section."""
    
    # First three sections: inputs, labels, meta_info
    inputs_section = display_inputs(datapoint['inputs'])
    labels_section = display_labels(datapoint['labels'])
    meta_info_section = display_meta_info(datapoint['meta_info'])
    
    # Fourth section: debug outputs (if present)
    debug_section = []
    if 'debug' in datapoint and datapoint['debug']:
        debug_section = [
            html.H3("Debug Outputs"),
            display_debug_outputs(datapoint['debug'])
        ]
    
    return html.Div([
        inputs_section,
        labels_section,
        meta_info_section,
        *debug_section  # Only included if debug outputs exist
    ])
```

## Configuration Example

```python
config = {
    # ... existing config ...
    
    'checkpoint_method': 5,  # Save checkpoints every 5 epochs
    
    'debugger': {
        'class': SequentialDebugger,
        'args': {
            'page_size_mb': 100,  # Save when buffer reaches 100MB
            'debuggers_config': [
                {
                    'name': 'layer4_features',
                    'debugger_config': {
                        'class': MyCustomFeatureDebugger,  # User-defined
                        'args': {
                            'layer_name': 'backbone.layer4'
                        }
                    }
                },
                {
                    'name': 'attention_maps',
                    'debugger_config': {
                        'class': AttentionDebugger,  # User-defined
                        'args': {
                            'layer_name': 'transformer.encoder.layer.5'
                        }
                    }
                },
                {
                    'name': 'gradcam',
                    'debugger_config': {
                        'class': GradCAMDebugger,  # User-defined
                        'args': {
                            'target_layer': 'backbone.layer4'
                        }
                    }
                }
            ]
        }
    }
}
```

## Usage Example

### 1. Define Custom Debuggers
```python
# User-defined debugger for feature extraction
class FeatureMapDebugger(ForwardDebugger):
    def process_forward(self, module, input, output):
        # Extract and process feature maps
        if isinstance(output, torch.Tensor):
            return {
                'feature_map': output.detach().cpu(),
                'stats': {
                    'mean': float(output.mean()),
                    'std': float(output.std()),
                    'shape': output.shape
                }
            }
        return None

# User-defined post-forward debugger
class GradCAMDebugger(BaseDebugger):
    def __init__(self, target_layer: str):
        self.target_layer = target_layer
        
    def __call__(self, datapoint, model):
        # Compute GradCAM using datapoint and model
        # Has access to full model and can compute gradients
        gradcam = self.compute_gradcam(datapoint['inputs'], datapoint['outputs'], model)
        return {'gradcam': gradcam, 'target': self.target_layer}
```

### 2. Training with Debug Outputs
```bash
python main.py --config-filepath configs/my_config_with_debugger.py
```

### 3. Viewing Debug Outputs
The eval viewer will automatically detect and display debug outputs for epochs where they were saved. Debug outputs appear as a fourth section below inputs, labels, and meta_info when viewing datapoints from checkpoint epochs.

## File Structure Example

```
work_dir/
├── epoch_4/
│   ├── checkpoint.pt
│   ├── training_losses.pt
│   ├── validation_scores.json
│   └── debugger/
│       ├── page_0.pkl  # List of (idx, debug_outputs) tuples
│       ├── page_1.pkl
│       └── page_2.pkl
├── epoch_9/
│   ├── checkpoint.pt
│   ├── training_losses.pt
│   ├── validation_scores.json
│   └── debugger/
│       └── page_0.pkl
└── epoch_14/  # Last epoch
    ├── checkpoint.pt
    ├── training_losses.pt
    ├── validation_scores.json
    └── debugger/
        └── page_0.pkl
```

## Key Design Principles

1. **Flexibility**: Users define their own debuggers for project-specific needs
2. **Integration**: Follows Pylon's existing patterns (criterion, metric, optimizer)
3. **Efficiency**: Page-based async saving prevents memory issues
4. **Simplicity**: Minimal changes to existing codebase
5. **Extensibility**: Easy to add new debugger types

## Benefits

1. **Pattern Identification**: Visualize model internals during failed predictions
2. **Research Insights**: Understand where algorithms struggle
3. **Debugging Power**: Access to both forward hooks and post-forward analysis
4. **Memory Efficient**: Page-based saving with configurable size limits
5. **Selective Saving**: Only saves at checkpoint epochs to manage storage
