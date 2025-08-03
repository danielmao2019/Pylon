# PARENet Integration Implementation Plan - Step 3: API Wrapper Creation

## Overview
Create thin wrapper layers to adapt PARENet's external code to Pylon's API requirements while preserving original functionality.

## Key Pylon Patterns Identified

### Model API Pattern
- Models accept `Dict[str, Any]` inputs and return `Dict[str, torch.Tensor]` outputs
- Use `build_from_config` pattern for initialization
- Follow standard model registration in `__init__.py` files

### Criterion API Pattern
- Inherit from `BaseCriterion` 
- Use `__call__(y_pred: Dict, y_true: Dict) -> torch.Tensor`
- Implement asynchronous buffer management for training
- Define `DIRECTIONS` attribute for loss types
- Implement `summarize()` method

### Metric API Pattern
- Inherit from `BaseMetric` or `SingleTaskMetric`
- Use `__call__(datapoint: Dict) -> Dict` pattern
- **Critical**: Must define `DIRECTIONS` attribute (1 for higher-is-better, -1 for lower-is-better)
- Support per-datapoint and aggregated scoring
- Use existing Pylon metrics where possible

### Collator API Pattern  
- Inherit from `BaseCollator`
- Handle nested dictionary collation
- Support custom collation functions per key
- Integrate with PARENet's stack-mode operations

## Implementation Tasks

### 1. Model Wrapper (`models/point_cloud_registration/parenet/parenet_model.py`)
**Goal**: Wrap `PARE_Net` to follow Pylon's model conventions

**Key Changes**:
- Rename original `PARE_Net` to `_PARE_Net` (make private)
- Create new `PARENetModel` class as public API
- Implement `build_from_config` support
- Handle input/output format conversion
- Ensure compatibility with Pylon's training pipeline

**Input Format**: Standard Pylon collated dict from PARENet collator
**Output Format**: 
```python
{
    'estimated_transform': torch.Tensor,  # For IsotropicTransformError metric
    'ref_corr_points': torch.Tensor,      # For InlierRatio metric  
    'src_corr_points': torch.Tensor,      # For InlierRatio metric
    'coarse_precision': torch.Tensor,     # For PARENet-specific metrics
    'fine_precision': torch.Tensor,       # For PARENet-specific metrics
}
```

### 2. Criterion Wrapper (`criteria/vision_3d/point_cloud_registration/parenet_criterion/parenet_criterion.py`)
**Goal**: Wrap `OverallLoss` to follow Pylon's `BaseCriterion` pattern

**Key Changes**:
- Rename original `OverallLoss` to `_OverallLoss` (make private)
- Create `PARENetCriterion(BaseCriterion)` class
- Handle the multi-component loss (coarse + fine RI + fine RE)
- Implement proper buffer management
- Define `DIRECTIONS = {"loss": -1, "c_loss": -1, "f_ri_loss": -1, "f_re_loss": -1}`

**API**: `__call__(y_pred: Dict, y_true: Dict) -> torch.Tensor`

### 3. Metric Wrapper (`metrics/vision_3d/point_cloud_registration/parenet_metric/`)
**Goal**: Create Pylon-compatible metrics for PARENet evaluation

**Components**:
- `parenet_metric.py`: Main wrapper combining multiple metrics
- Use existing Pylon metrics:
  - `IsotropicTransformError` for RRE/RTE
  - `InlierRatio` for IR/PIR
- Add PARENet-specific metrics (RMSE, RR)

**DIRECTIONS**:
```python
DIRECTIONS = {
    "rotation_error": -1,    # RRE - lower is better
    "translation_error": -1, # RTE - lower is better  
    "inlier_ratio": 1,       # IR - higher is better
    "point_inlier_ratio": 1, # PIR - higher is better
    "rmse": -1,              # RMSE - lower is better
    "registration_recall": 1  # RR - higher is better
}
```

### 4. Collator Integration (`data/collators/parenet/parenet_collator.py`)
**Goal**: Integrate PARENet's stack-mode collation with Pylon's `BaseCollator`

**Key Changes**:
- Create `PARENetCollator(BaseCollator)` class
- Integrate `registration_collate_fn_stack_mode` functionality
- Handle hierarchical point cloud processing
- Ensure compatibility with multi-GPU training
- Support both precomputed and on-demand neighbor computation

### 5. Configuration Integration
**Goal**: Create configuration builders for `build_from_config` pattern

**Files**:
- Update `models/point_cloud_registration/__init__.py`
- Update `criteria/vision_3d/point_cloud_registration/__init__.py` 
- Update `metrics/vision_3d/point_cloud_registration/__init__.py`
- Update `data/collators/__init__.py`

## Testing Strategy

### Model Tests (`tests/models/point_cloud_registration/parenet/`)
- `test_parenet_model.py`: Initialization, forward pass, gradient flow, API compliance

### Criterion Tests (`tests/criteria/vision_3d/point_cloud_registration/parenet_criterion/`)
- `test_parenet_criterion.py`: Loss computation, buffer management, DIRECTIONS validation

### Metric Tests (`tests/metrics/vision_3d/point_cloud_registration/parenet_metric/`)  
- `test_parenet_metric.py`: Score computation, DIRECTIONS attribute, per-datapoint/aggregated format

### Collator Tests (`tests/data/collators/parenet/`)
- `test_parenet_collator.py`: Stack-mode collation, neighbor computation, batch handling

## Key Principles

1. **Preserve Original Logic**: No changes to core PARENet algorithms
2. **Thin Wrappers**: Minimal code that only handles API conversion
3. **Naming Strategy**: Original classes become private (`_Class`), wrappers use public names
4. **API Compliance**: Follow Pylon's established patterns exactly
5. **DIRECTIONS Requirement**: All metrics must define DIRECTIONS for early stopping

## Dependencies

- Step 2 (import fixes) must be completed
- All PARENet imports working correctly
- Understanding of PARENet's data flow and loss components
- Familiarity with Pylon's base classes and patterns

## Success Criteria

- All wrapper classes instantiate correctly via `build_from_config`
- Model wrapper produces correct output format for metrics
- Criterion wrapper computes multi-component losses correctly
- Metric wrapper provides proper DIRECTIONS and scoring
- Collator wrapper handles PARENet's stack-mode requirements
- All components integrate seamlessly with Pylon's training pipeline