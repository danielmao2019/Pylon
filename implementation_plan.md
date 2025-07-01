# Early Stopping Implementation Plan

## Codebase Analysis Summary

After thorough exploration of the Pylon codebase, I've identified the key components and requirements for implementing early stopping.

### Current Architecture

1. **Training Loop**: Implemented in `BaseTrainer` (`runners/base_trainer.py`)
   - Main training loop in `run()` method (lines 590-611)
   - Per-epoch training and validation in `_train_epoch_()` and `_val_epoch_()`
   - Validation scores stored in `validation_scores.json` with structure:
     ```json
     {
       "aggregated": {"mean_IoU": 0.5, "mean_f1": 0.7, ...},
       "per_datapoint": {...}
     }
     ```

2. **Issue Found**: `_find_best_checkpoint_()` method (line 470) expects a `'reduced'` key that doesn't exist
   - This is a bug in the current implementation
   - Validation scores only have `'aggregated'` and `'per_datapoint'` keys

3. **Configuration System**: Uses `build_from_config()` pattern for all components

## Early Stopping Implementation Strategy

### Phase 1: Fix Existing Issues
1. **Fix `_find_best_checkpoint_()` method**:
   - Replace hardcoded `'reduced'` key with configurable metric selection
   - Add `monitor_metric` parameter to config (e.g., `'mean_IoU'`, `'mean_f1'`)
   - Extract metric value from `scores['aggregated'][monitor_metric]`

### Phase 2: Implement Early Stopping Component
1. **Create `EarlyStopping` class**:
   - Location: `utils/training/early_stopping.py`
   - Features:
     - Track validation metric history
     - Configurable patience (epochs without improvement)
     - Configurable min_delta (minimum improvement threshold)
     - Support for both minimize and maximize metrics
     - Save/restore state for training resumption

2. **Configuration Parameters**:
   ```python
   early_stopping_config = {
       'enabled': True,
       'monitor_metric': 'mean_IoU',  # Which metric to monitor
       'mode': 'max',  # 'max' for metrics to maximize, 'min' for minimize
       'patience': 10,  # Number of epochs without improvement
       'min_delta': 0.001,  # Minimum improvement threshold
       'restore_best_weights': True,  # Load best weights when stopping
   }
   ```

### Phase 3: Integration with BaseTrainer
1. **Modify `BaseTrainer.__init__()`**:
   - Initialize early stopping component if enabled in config
   - Set up metric monitoring configuration

2. **Modify training loop (`run()` method)**:
   - After each validation epoch, check early stopping condition
   - If should stop: log message and break training loop
   - If restore_best_weights: load best checkpoint

3. **Update validation logic**:
   - Extract monitored metric from validation scores
   - Update early stopping tracker with current metric value

### Phase 4: Testing and Documentation
1. **Create comprehensive tests**:
   - Test early stopping with different metrics
   - Test patience and min_delta functionality
   - Test training resumption with early stopping state
   - Test integration with existing trainer classes

2. **Update configuration examples**:
   - Add early stopping to example configs
   - Document configuration parameters

## Detailed Implementation

### 1. EarlyStopping Class Design

```python
class EarlyStopping:
    def __init__(
        self,
        monitor_metric: str,
        mode: str = 'max',
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True
    ):
        # Implementation details below
```

**Key Features**:
- Thread-safe design following Pylon's patterns
- State persistence for training resumption  
- Configurable metric direction (maximize/minimize)
- Best weights tracking

### 2. BaseTrainer Integration Points

**Modified methods**:
1. `__init__()`: Initialize early stopping
2. `_init_components_()`: Set up early stopping with metric info
3. `run()`: Main training loop with early stopping checks
4. `_find_best_checkpoint_()`: Fix to use configurable metric
5. `_after_val_loop_()`: Update early stopping tracker

**New methods**:
1. `_init_early_stopping_()`: Initialize early stopping component
2. `_should_stop_early()`: Check early stopping condition
3. `_handle_early_stop()`: Handle early stopping logic

### 3. Configuration Integration

**Required config keys**:
```python
config = {
    # Existing keys...
    'monitor_metric': 'mean_IoU',  # Fix for _find_best_checkpoint_
    'early_stopping': {
        'enabled': True,
        'monitor_metric': 'mean_IoU',  # Can differ from above
        'mode': 'max',
        'patience': 10,
        'min_delta': 0.001,
        'restore_best_weights': True,
    }
}
```

### 4. Backward Compatibility

- Early stopping is opt-in (disabled by default)
- Existing configs continue to work unchanged
- `monitor_metric` has sensible defaults per task type
- Graceful fallbacks for missing configuration

## Implementation Timeline

1. **Phase 1** (Fix existing issues): 1-2 hours
   - Fix `_find_best_checkpoint_()` method
   - Add `monitor_metric` configuration support

2. **Phase 2** (Early stopping component): 2-3 hours  
   - Implement `EarlyStopping` class
   - Add comprehensive tests

3. **Phase 3** (Integration): 2-3 hours
   - Integrate with `BaseTrainer`
   - Update all trainer subclasses if needed

4. **Phase 4** (Testing & docs): 1-2 hours
   - Integration testing
   - Update documentation and examples

**Total estimated time**: 6-10 hours

## Questions for Discussion

1. **Default monitor metrics**: Should we have task-specific defaults?
   - Change detection: `mean_IoU`
   - Classification: `accuracy` 
   - Multi-task: `reduced` (computed from multiple metrics)

2. **Early stopping placement**: Should it be in `BaseTrainer` or a separate component that trainers can optionally use?

3. **Metric direction inference**: Should we auto-detect if metric should be maximized/minimized based on metric name patterns?

4. **State persistence**: How detailed should early stopping state be in checkpoints?

5. **Multi-task learning**: How should early stopping work when multiple metrics are being tracked?

This plan provides a robust, backward-compatible early stopping implementation that follows Pylon's design patterns and conventions.
