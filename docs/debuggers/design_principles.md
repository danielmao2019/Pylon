# Debugger Design Principles

This document outlines the core design principles that guide the Pylon debuggers module architecture and implementation.

## 1. Flexibility

**Users define their own debuggers for project-specific needs**

- **Extensible base classes**: BaseDebugger and ForwardDebugger provide minimal interfaces
- **Custom implementations**: Researchers implement domain-specific debugging logic
- **Modular architecture**: Mix and match different debugger types in one configuration
- **No assumptions**: Framework doesn't assume what users want to debug

Example:
```python
class MyCustomDebugger(BaseDebugger):
    def __call__(self, datapoint, model):
        # Implement any custom debugging logic
        return custom_analysis(datapoint, model)
```

## 2. Integration

**Follows Pylon's existing patterns (criterion, metric, optimizer)**

- **Consistent APIs**: Same configuration and builder patterns as other components
- **Async buffer pattern**: Uses threading and queues like base_criterion.py and base_metric.py
- **Memory management**: Follows apply_tensor_op pattern for GPU to CPU conversion
- **Hook registration**: Automatic forward hook setup following PyTorch best practices

Integration examples:
```python
# Same config pattern as metrics/criteria
'debugger': {
    'class': SequentialDebugger,
    'args': {...}
}

# Same async buffer pattern
self._buffer_thread = threading.Thread(target=self._buffer_worker, daemon=True)
self._buffer_queue = queue.Queue()
```

## 3. Efficiency

**Memory-efficient with async saving**

### Memory Management
- **Page-based storage**: Configurable size limits prevent memory overflow
- **Background processing**: Non-blocking buffer operations during training
- **CPU conversion**: Use apply_tensor_op for efficient GPU → CPU tensor transfer
- **Deep size calculation**: Accurate memory tracking for intelligent page splitting

### Performance Optimizations
- **Lazy evaluation**: Only compute debug outputs when enabled
- **Selective saving**: Only save at checkpoint epochs during training
- **Thread safety**: Proper locking prevents data races without blocking
- **Efficient serialization**: Joblib for fast page saving/loading

```python
# Memory-efficient tensor handling
processed_outputs = apply_tensor_op(
    func=lambda x: x.detach().cpu(), 
    inputs=debug_outputs
)

# Page-based memory management
if self.current_page_size >= self.page_size:
    self._save_current_page()
```

## 4. Simplicity

**Minimal changes to existing codebase**

### Clean Integration Points
- **Single initialization call**: `_init_debugger()` after model setup
- **Single evaluation call**: `debugger(datapoint, model)` in eval step
- **Automatic enable/disable**: Based on checkpoint indices for training
- **Optional configuration**: System works without debugger config

### User-Friendly APIs
- **Simple inheritance**: Implement one method for custom debuggers
- **Clear configuration**: Descriptive parameter names and validation
- **Automatic hook management**: No manual hook registration required
- **Error handling**: Graceful degradation when debuggers fail

```python
# Minimal code changes required
if self.debugger and self.debugger.enabled:
    dp['debug'] = self.debugger(dp, self.model)
```

## 5. Extensibility

**Easy to add new debugger types**

### Pluggable Architecture
- **Abstract base classes**: Well-defined interfaces for new debugger types
- **Configuration-driven**: Add new debuggers through config, not code changes
- **Composable design**: Combine multiple debuggers in SequentialDebugger
- **Hook flexibility**: Support both forward hooks and post-processing

### Future-Proof Design
- **Version compatibility**: Stable interfaces that won't break with updates
- **Framework agnostic**: Core concepts work with different ML frameworks
- **Scalable patterns**: Architecture supports complex debugging workflows
- **Documentation-driven**: Clear examples for implementing new types

```python
# Easy to add new debugger types
class NewDebuggerType(BaseDebugger):
    def __call__(self, datapoint, model):
        # Implement new debugging approach
        return new_analysis()

# Automatic integration through config
'debuggers_config': [
    {
        'name': 'new_analysis',
        'debugger_config': {
            'class': NewDebuggerType,
            'args': {...}
        }
    }
]
```

## Benefits Derived from These Principles

### 1. Pattern Identification
- **Visual debugging**: See model internals during failed predictions
- **Comparative analysis**: Compare successful vs failed cases side-by-side
- **Temporal patterns**: Track how debugging outputs change across epochs

### 2. Research Insights
- **Failure analysis**: Understand where algorithms struggle
- **Hypothesis testing**: Validate research assumptions with visual evidence
- **Model interpretability**: Make deep learning models more transparent

### 3. Debugging Power
- **Multi-modal analysis**: Both forward hooks and post-forward processing
- **Full model access**: Complete datapoint and model available for analysis
- **Flexible timing**: Debug at any point in training or evaluation pipeline

### 4. Production Ready
- **Memory efficient**: Won't crash on large datasets or long training runs
- **Storage optimized**: Only saves when needed, configurable size limits
- **Thread safe**: Robust concurrent access patterns
- **Error resilient**: Continues training even if debugging fails

### 5. Developer Experience
- **Easy to use**: Minimal learning curve for researchers
- **Well documented**: Comprehensive examples and integration guides
- **Testable**: 79 tests ensure reliability and correctness
- **Maintainable**: Clean code following established patterns

## Design Trade-offs

### Flexibility vs Performance
- **Choice**: Prioritized flexibility over maximum performance
- **Rationale**: Research needs vary greatly, customization more valuable than speed
- **Mitigation**: Async processing and selective saving minimize performance impact

### Simplicity vs Features  
- **Choice**: Kept core interfaces simple, advanced features in implementations
- **Rationale**: Lower barrier to entry, advanced users can build complex debuggers
- **Mitigation**: Rich examples show how to implement advanced features

### Memory vs Storage
- **Choice**: Trade memory usage for storage efficiency through page-based saving
- **Rationale**: Training runs are memory-constrained, storage is cheaper
- **Mitigation**: Configurable page sizes and selective saving

These design principles ensure the debuggers module serves both immediate debugging needs and long-term research goals while maintaining compatibility with Pylon's architecture and philosophy.
