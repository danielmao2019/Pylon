# Explicit Metric Directions Design

## Overview

The metric direction system has been redesigned to use explicit `DIRECTIONS` attributes instead of magical inference. Each metric now explicitly defines directions for all score keys it produces.

## Design Principles

1. **Explicit over Implicit**: No magic code inference - each metric declares its own directions
2. **Accurate**: Directions match actual output keys that metrics produce
3. **Maintainable**: Each metric owns its direction definition
4. **Type-safe**: No runtime inspection required

## Implementation Examples

### Single Metrics

```python
class SemanticSegmentationMetric(SingleTaskMetric):
    # Define explicit directions for all score keys this metric produces
    DIRECTIONS = {
        'mean_IoU': 1,          # Higher is better
        'accuracy': 1,          # Higher is better  
        'mean_precision': 1,    # Higher is better
        'mean_recall': 1,       # Higher is better
        'mean_f1': 1,          # Higher is better
    }
```

### Complex Metrics (Nested Structure)

```python
class ChangeStarMetric(SingleTaskMetric):
    # Define explicit directions matching exact output structure
    # Each task has its own nested set of metric directions
    DIRECTIONS = {
        'change': SemanticSegmentationMetric.DIRECTIONS,      # Inherits from change_metric
        'semantic_1': SemanticSegmentationMetric.DIRECTIONS, # Inherits from semantic_metric  
        'semantic_2': SemanticSegmentationMetric.DIRECTIONS, # Inherits from semantic_metric
    }
    # Results in:
    # {
    #     'change': {'mean_IoU': 1, 'accuracy': 1, 'mean_precision': 1, 'mean_recall': 1, 'mean_f1': 1},
    #     'semantic_1': {'mean_IoU': 1, 'accuracy': 1, 'mean_precision': 1, 'mean_recall': 1, 'mean_f1': 1},
    #     'semantic_2': {'mean_IoU': 1, 'accuracy': 1, 'mean_precision': 1, 'mean_recall': 1, 'mean_f1': 1}
    # }
```

### Multi-Task Metrics (Dynamic DIRECTIONS)

```python
class MultiTaskMetric(BaseMetric):
    # NOTE: DIRECTIONS cannot be class attribute for wrapper metrics
    # because they depend on runtime configuration of component metrics
    
    def __init__(self, metric_configs: dict) -> None:
        self.task_metrics = {
            task: build_from_config(config=metric_configs[task])
            for task in metric_configs.keys()
        }
        
        # Build DIRECTIONS from component task metrics (instance attribute)
        self.DIRECTIONS = {}
        for task_name, task_metric in self.task_metrics.items():
            if hasattr(task_metric, 'DIRECTIONS'):
                # Use first direction (assumes all scores in task have same direction)
                self.DIRECTIONS[task_name] = list(task_metric.DIRECTIONS.values())[0]
            elif hasattr(task_metric, 'DIRECTION'):
                # Legacy fallback
                self.DIRECTIONS[task_name] = task_metric.DIRECTION
```

### Hybrid Metrics (Dynamic DIRECTIONS)

```python
class HybridMetric(SingleTaskMetric):
    # NOTE: DIRECTIONS cannot be class attribute for wrapper metrics
    # because they depend on runtime configuration of component metrics
    
    def __init__(self, metrics_cfg: List[Dict]) -> None:
        # Build component metrics...
        
        # Build DIRECTIONS by merging all component directions (instance attribute)
        self.DIRECTIONS = {}
        for component_metric in self.metrics:
            if hasattr(component_metric, 'DIRECTIONS'):
                # Merge all score keys from component
                self.DIRECTIONS.update(component_metric.DIRECTIONS)
            elif hasattr(component_metric, 'DIRECTION'):
                # Legacy fallback
                self.DIRECTIONS[f"metric_{i}"] = component_metric.DIRECTION
```

## API Usage

```python
from runners.model_comparison import get_metric_directions

# Single metric with class attribute - can access without instantiation
directions = SemanticSegmentationMetric.DIRECTIONS
# Returns: {'mean_IoU': 1, 'accuracy': 1, 'mean_precision': 1, 'mean_recall': 1, 'mean_f1': 1}

# Or access from instance
metric = SemanticSegmentationMetric(num_classes=10)
directions = get_metric_directions(metric)
# Returns same result

# Complex metric with nested class attributes
nested_directions = ChangeStarMetric.DIRECTIONS  
# Returns nested structure matching output

# Dynamic wrapper metrics - MUST use instances (no class attribute)
task_configs = {'seg': {...}, 'det': {...}}
metric = MultiTaskMetric(task_configs)
directions = get_metric_directions(metric)  # Instance required
# Returns: {'seg': 1, 'det': 1}

# Use in early stopping
early_stopping = EarlyStopping(metric=metric, ...)
# Automatically uses metric.DIRECTIONS for score comparison
```

## Migration Guide

### Before (Magic Inference)
```python
class MyMetric(SingleTaskMetric):
    DIRECTION = 1  # Applied to all scores magically
```

### After (Explicit Declaration)
```python
class MyMetric(SingleTaskMetric):
    DIRECTIONS = {
        'accuracy': 1,     # Explicitly list each score key
        'f1_score': 1,     # that this metric produces
        'precision': 1,
    }
```

## Class vs Instance DIRECTIONS

### When to Use Class Attributes
- **Fixed structure metrics**: Metrics with known, unchanging output structure
- **Simple metrics**: Single-task metrics with predefined score keys
- **Examples**: `SemanticSegmentationMetric`, `ChangeStarMetric`

```python
class SemanticSegmentationMetric(SingleTaskMetric):
    DIRECTIONS = {'mean_IoU': 1, 'accuracy': 1, ...}  # Class attribute
```

### When to Use Instance Attributes  
- **Dynamic wrapper metrics**: Metrics whose structure depends on configuration
- **Configurable metrics**: Multi-task or hybrid metrics with variable components
- **Examples**: `MultiTaskMetric`, `HybridMetric`

```python
class MultiTaskMetric(BaseMetric):
    def __init__(self, metric_configs):
        # Build DIRECTIONS based on provided task configs
        self.DIRECTIONS = {...}  # Instance attribute
```

## Benefits

1. **Clear Intent**: Each metric explicitly states what scores it produces and their directions
2. **Type Safety**: No runtime errors from missing direction mappings
3. **Maintainability**: Easy to see what scores a metric produces
4. **Accuracy**: Directions match actual output keys, no mismatches
5. **Debuggability**: Clear error messages when directions are missing
6. **Flexible Access**: Class attributes when possible, instance attributes when necessary

## Error Handling

```python
# Missing DIRECTIONS attribute
class BadMetric:
    pass

get_metric_directions(BadMetric())
# Raises: AttributeError: Metric <class 'BadMetric'> has no DIRECTIONS attribute

# Invalid direction values  
class InvalidMetric:
    DIRECTIONS = {"score": 0}  # Invalid!

get_metric_directions(InvalidMetric())
# Raises: AssertionError: DIRECTION for 'score' must be -1 or 1, got 0
```
