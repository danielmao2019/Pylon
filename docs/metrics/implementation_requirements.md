# Metric Structure Requirements

## Critical Requirement: Matching Keys Between Aggregated and Per-Datapoint Results

### Overview

**ALL metrics in Pylon must ensure that `scores['per_datapoint']` and `scores['aggregated']` have exactly matching keys.** This requirement is enforced by the evaluation viewer backend and violating it will cause assertion failures.

### Technical Details

The requirement is enforced in `/runners/eval_viewer/backend/initialization.py`:

```python
# Lines 161 and 204
metric_names = get_metric_names_aggregated(scores['aggregated'])
assert metric_names == get_metric_names_per_datapoint(scores['per_datapoint'])
```

### Implementation Pattern

**✅ CORRECT - Matching Keys:**
```python
def summarize(self) -> Dict[str, Dict[str, torch.Tensor]]:
    # Calculate results
    individual_scores = torch.stack([score["mse"] for score in self.buffer])
    average_score = individual_scores.mean()
    
    return {
        "aggregated": {
            "mse": average_score,
            "rmse": torch.sqrt(average_score)
        },
        "per_datapoint": {
            "mse": individual_scores,
            "rmse": torch.sqrt(individual_scores)  # Same keys as aggregated!
        }
    }
```

**❌ WRONG - Mismatched Keys:**
```python
def summarize(self) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        "aggregated": {
            "mse": average_score,
            "rmse": torch.sqrt(average_score)
        },
        "per_datapoint": {
            "mse": individual_scores
            # Missing "rmse" key - WILL FAIL!
        }
    }
```

### Why This Requirement Exists

1. **Evaluation Viewer**: The web-based evaluation viewer expects consistent metric names across both sections
2. **Data Analysis**: Tools expect to be able to correlate per-datapoint and aggregated results using the same keys
3. **Consistency**: Ensures uniform metric reporting across the entire framework

### Common Mistakes

1. **Forgetting to include all keys**: Adding a metric to aggregated but not per_datapoint
2. **Different key names**: Using "accuracy" in aggregated but "acc" in per_datapoint  
3. **Empty case handling**: Not maintaining key consistency when buffer is empty (though this should never happen - see NO DEFENSIVE PROGRAMMING principle)

### Testing Your Metrics

Always test your metric implementations to ensure key consistency:

```python
def test_metric_key_consistency():
    metric = YourCustomMetric()
    # ... add some data to metric buffer ...
    result = metric.summarize()
    
    # Ensure keys match exactly
    aggregated_keys = set(result['aggregated'].keys())
    per_datapoint_keys = set(result['per_datapoint'].keys())
    assert aggregated_keys == per_datapoint_keys, f"Key mismatch: {aggregated_keys} vs {per_datapoint_keys}"
```

### Framework Examples

All built-in metrics follow this pattern:
- `SemanticSegmentationMetric`
- `ConfusionMatrix`
- `ObjectDetectionMetric`
- `TransformInlierRatio`

Refer to these implementations for guidance when creating custom metrics.
