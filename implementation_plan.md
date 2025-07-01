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

### Phase 1: Early Stopping Logic (Priority 1)
1. **Add early stopping to BaseTrainer**:
   - Track validation score history for partial order comparison
   - Implement early stopping check after each validation epoch
   - Handle "not improving" case for incomparable vectors
   - Stop training when no improvement for specified epochs

2. **Fix `_init_state_()` in runners module**:
   - Update logic to handle early stopping completion
   - Redefine what constitutes a "completed" run
   - Check for early stopping markers in addition to epoch count

3. **Configuration system**:
   ```python
   config = {
       'order': False,  # Default: no reduction (vector comparison)
       # OR
       'order': True,   # Average all scores
       # OR  
       'order': {       # Weighted combination (same structure as scores['aggregated'])
           'mean_IoU': 0.7,
           'mean_f1': 0.3,
           # Missing scores default to 0.0
       },
       
       'early_stopping': {
           'enabled': True,  # Default True if config exists
           'epochs': 10,     # Patience epochs without improvement
       }
   }
   ```

### Phase 2: Best Checkpoint Logic (Priority 2)
1. **Fix `_find_best_checkpoint()` method** (rename from `_find_best_checkpoint_()`):
   - Replace hardcoded `'reduced'` key with `config['order']` system
   - Handle incomparable vectors by falling back to equal-weight average
   - Extract scores from `scores['aggregated']` structure only
   - Support all three comparison modes

2. **Score comparison utilities**:
   - Location: `utils/training/score_comparison.py`
   - Vector comparison with first quadrant cone partial order
   - Weighted averaging with normalization
   - Fallback to equal-weight average for incomparable cases

### Phase 3: Agents Module (Separate PR)
1. **Deferred to separate task**:
   - Update progress checking in `agents` module
   - This will be handled in another PR

## Detailed Implementation

### 1. Early Stopping Implementation

```python
# In BaseTrainer class
def _should_stop_early(self, current_epoch_scores: Dict[str, Any]) -> bool:
    """
    Check if training should stop early based on score history.
    
    Uses vector comparison (partial order) for improvement detection.
    Incomparable vectors are treated as "not improving".
    """
```

**Key Features**:
- Vector comparison using first quadrant cone partial order
- Track score history across epochs
- "Not improving" for incomparable cases
- Integration with existing validation loop

### 2. Score Comparison System

```python
# utils/training/score_comparison.py
def compare_scores_vector(
    current_scores: Dict[str, Any],
    best_scores: Dict[str, Any], 
    metric_directions: Dict[str, int]
) -> Optional[bool]:
    """
    Vector comparison using partial order.
    Returns None for incomparable cases.
    """

def reduce_scores_to_scalar(
    scores: Dict[str, Any],
    order_config: Union[bool, Dict],
    metric_directions: Dict[str, int]
) -> float:
    """
    Reduce scores to scalar for best checkpoint selection.
    Handles weighted and equal-weight averaging.
    """
```

### 3. BaseTrainer Integration Points

**Modified methods**:
1. `_init_state_()`: Handle early stopping completion detection
2. `run()`: Add early stopping check after validation epochs
3. `_find_best_checkpoint()`: Use `config['order']` with fallback averaging
4. `_after_val_loop_()`: Track score history for early stopping

**New methods**:
1. `_should_stop_early()`: Early stopping condition check
2. `_get_metric_directions()`: Extract DIRECTION from metric classes
3. `_compare_scores()`: Wrapper for score comparison utilities

### 4. Configuration Schema

**Updated structure (matching scores['aggregated'])**:
```python
config = {
    'order': False,  # Default: vector comparison, no reduction
    # OR
    'order': True,   # Equal-weight average  
    # OR
    'order': {       # Weights dict (same structure as scores['aggregated'])
        'mean_IoU': 0.7,
        'mean_f1': 0.3,
    },
    
    'early_stopping': {
        'enabled': True,   # Default True if section exists
        'epochs': 10,      # Patience
    }
}
```

### 5. Backward Compatibility

- `config['order']` defaults to `False` (no reduction) if not provided
- `config['early_stopping']` is optional - disabled if not present
- Existing configs work unchanged
- Multi-stage training uses separate configs per stage

## Implementation Timeline

1. **Phase 1** (Early stopping - Priority 1): 4-5 hours
   - Add early stopping logic to BaseTrainer training loop
   - Implement vector comparison for early stopping (partial order)
   - Fix `_init_state_()` to handle early stopping completion
   - Add score history tracking

2. **Phase 2** (Best checkpoint - Priority 2): 3-4 hours  
   - Fix `_find_best_checkpoint()` to use `config['order']`
   - Implement score reduction utilities in `utils/training/score_comparison.py`
   - Handle fallback to equal-weight average for incomparable vectors
   - Extract metric DIRECTION attributes

3. **Phase 3** (Testing & validation): 2-3 hours
   - Add comprehensive tests for both early stopping and checkpoint logic
   - Test configuration parsing and defaults
   - Validate backward compatibility with existing configs

4. **Phase 4** (Agents module): SEPARATE PR
   - Update agents module progress checking
   - Will be handled in separate pull request

**Total estimated time**: 9-12 hours (this PR), agents module in separate PR

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

### Answers (Round 2)

1. When using the partial order in high-dimensional Euclidean space for early stopping, you only need to know when the model is improving and when it is not improving. For the incomparable case, it is "not improving". For finding the best checkpoint, yes there is a problem, because you must be able to compare any two models. So in the case that you are not able to find a model that is better than all other models, then you use the equal-weight average to reduce the high-dimensional scores to a scalar.

2. You should assume that all epochs have the same scores structure, because they will be using the same metric class. The only case is multi-stage training. But in this case you need a separate config for each stage, then you have the opportunity to provide different order configs if the scores structure changes. Note that for model comparison, you don't need the `per_datapoint` field in the scores. You only look at the `aggregated` field of the scores dict. The `order` field should have the same structure as the dict `scores['aggregated']` (not the same structure as the dict `scores`).

3. Do "Add early stopping logic" first, including fixing the `_init_state` in the runners module. that's the first thing. Then you fix `_find_best_checkpoint_` (please rename to `_find_best_checkpoint`). Lastly you fix other modules like `agents`.

4. When `config['order']` is not provided, default to `False`, meaning that you don't do any reduction.

5. Let's do this task in another PR.
