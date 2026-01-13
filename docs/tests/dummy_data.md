# Dummy Data Generation for Tests

## Standardized Dummy Data Generators

**For common test cases, you can use the standardized dummy data generators** in `tests/models/utils/dummy_data_generators.py`:

**Note**: These generators are provided for convenience when testing common data types. However, **test suites should use their own specific dummy data** when they need to test something specific or have particular requirements.

```python
# Correct - uses proper tensor types
from tests.models.utils.dummy_data_generators import (
    generate_change_detection_data,      # (N, 3, H, W) float32 images
    generate_segmentation_labels,        # (N, H, W) int64 labels
    generate_classification_labels,      # (N,) int64 labels
    generate_point_cloud_data,          # Batched format for registration
    generate_point_cloud_segmentation_data,  # Flattened format for segmentation
)

# Test with proper types
images = generate_change_detection_data(batch_size=2, height=64, width=64)
labels = generate_segmentation_labels(batch_size=2, height=64, width=64, num_classes=10)
```

## Creating Custom Dummy Data

When creating test-specific dummy data, **always specify proper dtypes:**
```python
# Wrong - missing dtype specification
torch.randn(2, 3, 224, 224)                     # Should be dtype=torch.float32
torch.randint(0, 10, (2, 224, 224))            # Should be dtype=torch.int64

# Correct - use generators or specify dtypes  
torch.randn(2, 3, 224, 224, dtype=torch.float32)
torch.randint(0, 10, (2, 224, 224), dtype=torch.int64)
```

## Tensor Type Requirements

**CRITICAL for testing**: When generating dummy inputs in `tests/` modules, always follow the standardized tensor type conventions.

**For complete tensor type requirements, see `@docs/tensor_conventions.md`.**

The framework enforces strict tensor type conventions for:
- Images (float32, specific channel ordering)  
- Segmentation masks (int64, specific shapes)
- Classification labels (int64 scalars/tensors)
- Point clouds (use `PointCloud` objects with `xyz` coordinates)
- Model predictions (task-specific formats)

**Key principle**: Always specify `dtype` when creating test tensors to match the framework requirements.

### Point Cloud Test Inputs

Tests that need point clouds should construct `PointCloud` instances from `data.structures.three_d.point_cloud.point_cloud` rather than pretending with ad-hoc dictionaries. `PointCloud` enforces the expected coordinate shape and dtype, keeps all fields on the same device, and exposes `num_points` so downstream logic never has to look at `.xyz.shape[0]`.

```python
from data.structures.three_d.point_cloud.point_cloud import PointCloud

pc = PointCloud(xyz=torch.randn(1024, 3, dtype=torch.float32))
pc.feat = torch.ones(pc.num_points, 1, dtype=torch.float32)
assert pc.num_points == 1024
```
