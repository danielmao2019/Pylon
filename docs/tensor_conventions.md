# Tensor Type Conventions

**Standardized tensor formats across the Pylon framework.**

Pylon enforces strict tensor type conventions validated by `utils/input_checks/` module. These conventions apply to all parts of the framework: datasets, models, metrics, criteria, and tests.

## Framework-Wide Tensor Requirements

### Images
`(C, H, W)` float32 tensors where C=3, batched: `(N, C, H, W)`

```python
# Individual image
image = torch.randn(3, 224, 224, dtype=torch.float32)
# Batched images  
images = torch.randn(8, 3, 224, 224, dtype=torch.float32)
```

### Segmentation Masks
`(H, W)` int64 tensors, batched: `(N, H, W)`

```python
# Individual mask (semantic, binary, amodal, instance)
mask = torch.randint(0, num_classes, (224, 224), dtype=torch.int64)
# Batched masks
masks = torch.randint(0, num_classes, (8, 224, 224), dtype=torch.int64)
```

### Classification Labels
int64 scalars, batched: `(N,)` int64 tensors

```python
# Individual label
label = torch.tensor(5, dtype=torch.int64)
# Batched labels
labels = torch.randint(0, num_classes, (8,), dtype=torch.int64)
```

### Point Clouds
Point clouds are represented by the `PointCloud` class found in `data.structures.three_d.point_cloud.point_cloud`.
The class ensures that `xyz` is a `(N, 3)` float tensor without NaNs or Infs and exposes `num_points` for length checks.

Because the constructor already asserts these invariants, the rest of the pipeline can
treat each `PointCloud` instance as the canonical container. Attach optional fields,
neighbor indices, or pooling metadata directly to the object (for instance through
attributes or the `.data` dictionary) instead of building ad hoc dicts of tensors.
Passing the validated `PointCloud` into datasets, collators, and models keeps the
representation consistent and removes the need for legacy dict-level guards.

```python
from data.structures.three_d.point_cloud.point_cloud import PointCloud

pc = PointCloud(xyz=torch.randn(1024, 3, dtype=torch.float32))
# Optional fields can be attached via attributes (e.g., `pc.rgb = torch.randn(pc.num_points, 3)`)
batch = PointCloud(xyz=torch.randn(2048, 3, dtype=torch.float32))
```

Avoid reading `pc.xyz.shape[0]` directly; prefer `pc.num_points`.

### Model Predictions
Follow task-specific formats

- **Classification**: `(N, C)` float32 for batched, `(C,)` for individual
- **Segmentation**: `(N, C, H, W)` float32 for batched, `(C, H, W)` for individual
- **Depth**: `(N, 1, H, W)` float32 for batched, `(1, H, W)` for individual

## Critical Notes

**ALWAYS specify dtype**: When creating tensors programmatically, always specify the `dtype` parameter to match these conventions.

```python
# ✅ CORRECT - Always specify dtype
torch.randn(2, 3, 224, 224, dtype=torch.float32)
torch.randint(0, 10, (2, 224, 224), dtype=torch.int64)

# ❌ WRONG - Missing dtype specification
torch.randn(2, 3, 224, 224)                     # Should be dtype=torch.float32
torch.randint(0, 10, (2, 224, 224))            # Should be dtype=torch.int64
```

**Input validation**: The framework's input validation will fail if these type assumptions are not followed.

**Testing compliance**: When generating dummy inputs in tests, always follow these type assumptions.
