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

## Q&A

### Questions (Round 1)

1. **Default monitor metrics**: Should we have task-specific defaults?
   - Change detection: `mean_IoU`
   - Classification: `accuracy` 
   - Multi-task: `reduced` (computed from multiple metrics)

2. **Early stopping placement**: Should it be in `BaseTrainer` or a separate component that trainers can optionally use?

3. **Metric direction inference**: Should we auto-detect if metric should be maximized/minimized based on metric name patterns?

4. **State persistence**: How detailed should early stopping state be in checkpoints?

5. **Multi-task learning**: How should early stopping work when multiple metrics are being tracked?

### Answers (Round 1)

I will answer your 5 questions from three perspectives.

First, regarding your "Default monitor metrics", "Metric direction inference", and "Multi-task learning" 3 questions, they are actually asking about the same thing - when given two models, how do we compare them and say that one is better than the other, or not. This is indeed the hardest part to implement early stopping. For Question 3, no. If you check the @metrics module, each concrete metric class will have a class attribute DIRECTION, where +1 means the higher the better, and -1 means the lower the better. For Question 5, Multi-Task Learning (MTL), there is the weighted average relative improvement to Single-Task Learning counter-parts. I'm not sure if you are able to find it in the code base. This is the standard in the field of MTL, so I'm not worried about it. We just stick with the existing literature. So that answers Question 5. However, even with this evaluation protocol for MTL, we still need to figure out what's the representative score for each individual task in MTL. e.g., we can have mIoU and pixel accuracy for segmentation task, and we can have accuracy and F1 for classification task. So all the headaches reduce to Question 1. I think we should give the user the option to customize how to define a partial order on the complex (nested) scores dict. The user can specify in the configs to use one of the scores, e.g., accuracy, to represent the "score", and use the usual partial order on the real line; or use a (weighted average) version of all the scores; you can think of picking only one score as a special case of weighted average of all scores; or use multiple scores simultaneously, as a vector, e.g., both accuracy and F1, to represent the "score", and use the partial order in high-dimensional Euclidean space, defined by the first quadrant cone; i.e., we say a vector is greater than or equal to another vector if and only if all it's entries are greater than or equal to the corresponding entries in the other vector. Note that when you are making comparisons, you have to multiply each of the scores with the DIRECTION attribute.

The conversation natually leads to the answer to Question 2 "Early stopping placement" - how should a user configure the early stopping mechanism, including how to turn it on and off. I think there should be an additional "early_stopping" field in the `config` dict, which itself is a dict. `config['early_stopping']` should have the following keys. (1) `'enabled'`: default to `True` if not provided. disregard all following fields if `False`. if the `config['early_stopping']` is not provided or is an empty dict, then treat as not enabled. (2) `'epochs'`: the count of epochs for which the model has not been improved before early stopping. There should also be another new filed `'order'` in the `config` dict: if `False` or `None`, then do not reduce and use the high-dimensional partial order (defined by the first quadrant cone). if `True` then use the average of all individual scores. otherwise, this must be a (nested) dict having the exact same structure as the output scores dict, but the leaf entries would be the weights, instead of the actual scores. this (nested) dict must not have any path to leaf that does not exist in the output scores dict (check using assertions), but can have fewer. in case that a score is not specified a weight, default the weight to 0.0. when the user is configuring the (nested) weights dict through this `'order'` field, the user does not have to make sure that the weights sum to 1. the code should be normalizing the weights (stored in nested dict) to sum to 1. the user is not allowed to provide negative weights (check using assertions). Extending beyond this question, with the `config['order']` new field, you can re-implement the logic to find the best model checkpoint and use that on the test set, if the test split exists.

Lastly, for Question 4 "State persistence", I don't think you need to modify the checkpoints in any way. There is also no early stopping state. However, with early stopping, you need to upgrade entirely the logic in the code base to define the "progress" of a run. Specifically how to define that a run has completed. Previously this is as simple as counting the number of completed epochs and compare against the predefined total number of epochs. Now you also need to take the `config['order']` to determine this. This will involve lots of refactoring and restructuring, including in the `_init_state` in the runners, and the progress check in `agents` module. So let's do this at the end.

### Questions (Round 2)

Based on your answers, I have some clarification questions:

1. **Vector comparison precedence**: When using vector comparison (order=False/None), how should we handle cases where vectors are incomparable (e.g., [0.8, 0.6] vs [0.7, 0.7])? Should we:
   - Keep current best if new is not strictly better?
   - Use some tie-breaking mechanism?
   - Require at least one score improvement?

2. **Metric structure access**: You mentioned the nested dict must match scores structure. Should I assume all metrics produce the same nested structure, or do I need to handle cases where different epochs might have different score keys?

3. **Implementation priority**: Should I tackle this in phases?
   - Phase 1: Fix `_find_best_checkpoint_()` with `config['order']`
   - Phase 2: Add early stopping logic
   - Phase 3: Refactor run completion logic in agents/runners
   
4. **Backward compatibility**: For existing configs without `config['order']`, should I:
   - Default to `order=True` (average all scores)?
   - Try to infer a sensible single metric based on task type?
   - Require explicit configuration?

5. **agents module scope**: The agents module refactoring sounds extensive. Should this be a separate task, or is it essential for basic early stopping functionality?

This plan provides a robust, backward-compatible early stopping implementation that follows Pylon's design patterns and conventions.
