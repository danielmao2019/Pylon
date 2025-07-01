# Debugger Integration Guide

This guide details how to integrate the debuggers module with BaseTrainer and BaseEvaluator, including code modifications and configuration patterns.

## BaseTrainer Integration

### 1. Initialization Sequence

The debugger is initialized after the model to ensure hooks can be registered:

```python
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
```

### 2. Checkpoint Index Management

The trainer precomputes when to save debug outputs based on checkpoint method:

```python
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
```

### 3. Debugger Initialization

```python
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

### 4. Validation Loop Integration

```python
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

def _eval_step(self, dp: Dict[str, Dict[str, Any]], flush_prefix: Optional[str] = None) -> None:
    # ... existing code through model forward and metric computation ...
    
    # Add debug outputs (only during validation/test at checkpoint indices)
    if self.debugger and self.debugger.enabled:
        dp['debug'] = self.debugger(dp, self.model)
    
    # ... rest of existing code ...

def _after_val_loop_(self):
    # ... existing saves ...
    
    # Save debugger outputs if enabled
    if self.debugger and self.debugger.enabled:
        debugger_dir = os.path.join(epoch_root, "debugger")
        self.debugger.save_all(debugger_dir)
```

## BaseEvaluator Integration

### 1. Initialization

Similar to BaseTrainer but simpler since evaluation always runs debuggers:

```python
def _init_debugger(self):
    """Initialize debugger for evaluation."""
    if self.config.get('debugger', None):
        self.debugger = build_from_config(self.config['debugger'])
        
        # Register forward hooks
        for layer_name, debuggers in self.debugger.forward_debuggers.items():
            layer = get_layer_by_name(self.model, layer_name)
            if layer is not None:
                for debugger in debuggers:
                    layer.register_forward_hook(debugger.forward_hook_fn)
    else:
        self.debugger = None
```

### 2. Evaluation Loop Integration

```python
def _eval_epoch_(self):
    # Enable debugger for all evaluation datapoints
    if self.debugger:
        self.debugger.enabled = True
        self.debugger.reset_buffer()
    
    # ... existing evaluation loop ...

def _eval_step(self, dp: Dict[str, Dict[str, Any]]) -> None:
    # ... existing code ...
    
    # Add debug outputs (always during evaluation)
    if self.debugger and self.debugger.enabled:
        dp['debug'] = self.debugger(dp, self.model)

def _after_eval_loop_(self):
    # ... existing saves ...
    
    # Save debugger outputs
    if self.debugger and self.debugger.enabled:
        debugger_dir = os.path.join(self.work_dir, "debugger")
        self.debugger.save_all(debugger_dir)
```

## Eval Viewer Integration

### 1. Loading Debug Outputs

```python
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

### 2. Display Integration

```python
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

## Configuration Patterns

### Complete Training Configuration

```python
config = {
    'checkpoint_method': 5,  # Save checkpoints every 5 epochs
    
    'debugger': {
        'class': SequentialDebugger,
        'args': {
            'page_size_mb': 100,  # Save when buffer reaches 100MB
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
                        'args': {'layer_name': 'transformer.encoder.layer.5'}
                    }
                },
                {
                    'name': 'gradcam_analysis',
                    'debugger_config': {
                        'class': GradCAMDebugger,
                        'args': {'target_layer': 'backbone.layer4'}
                    }
                }
            ]
        }
    }
}
```

### Evaluation-Only Configuration

```python
config = {
    'debugger': {
        'class': SequentialDebugger,
        'args': {
            'page_size_mb': 50,  # Smaller pages for evaluation
            'debuggers_config': [
                {
                    'name': 'failure_analysis',
                    'debugger_config': {
                        'class': FailureAnalysisDebugger,
                        'args': {'threshold': 0.5}
                    }
                }
            ]
        }
    }
}
```

## File Structure

### Training with Debug Outputs
```
work_dir/
├── epoch_4/
│   ├── checkpoint.pt
│   ├── validation_scores.json
│   └── debugger/
│       ├── page_0.pkl
│       ├── page_1.pkl
│       └── page_2.pkl
├── epoch_9/
│   ├── checkpoint.pt
│   ├── validation_scores.json
│   └── debugger/
│       └── page_0.pkl
└── epoch_14/  # Last epoch
    ├── checkpoint.pt
    ├── validation_scores.json
    └── debugger/
        └── page_0.pkl
```

### Evaluation with Debug Outputs
```
work_dir/
├── evaluation_scores.json
└── debugger/
    ├── page_0.pkl
    ├── page_1.pkl
    └── page_2.pkl
```

## Benefits

1. **Selective Saving**: Only saves at checkpoint epochs during training to manage storage
2. **Always Available**: All datapoints get debug outputs during evaluation
3. **Automatic Integration**: Minimal code changes required
4. **Flexible Configuration**: Easy to enable/disable and configure different debuggers
5. **Memory Efficient**: Page-based saving prevents memory issues during long training runs

## Usage Examples

### Training with Debug Outputs
```bash
python main.py --config-filepath configs/my_training_config_with_debugger.py
```

### Evaluation with Debug Outputs  
```bash
python eval.py --config-filepath configs/my_evaluation_config_with_debugger.py
```

### Viewing Results
The eval viewer automatically detects and displays debug outputs when available, showing them as a fourth section alongside inputs, labels, and meta_info.
