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

**CRITICAL for testing**: When generating dummy inputs in `tests/` modules, always follow these type assumptions. The input validation will fail otherwise.

### Standardized tensor formats across the framework:

- **Images**: `(C, H, W)` float32 tensors where C=3, batched: `(N, C, H, W)`
  ```python
  # Individual image
  image = torch.randn(3, 224, 224, dtype=torch.float32)
  # Batched images  
  images = torch.randn(8, 3, 224, 224, dtype=torch.float32)
  ```

- **Segmentation masks**: `(H, W)` int64 tensors, batched: `(N, H, W)`
  ```python
  # Individual mask (semantic, binary, amodal, instance)
  mask = torch.randint(0, num_classes, (224, 224), dtype=torch.int64)
  # Batched masks
  masks = torch.randint(0, num_classes, (8, 224, 224), dtype=torch.int64)
  ```

- **Classification labels**: int64 scalars, batched: `(N,)` int64 tensors
  ```python
  # Individual label
  label = torch.tensor(5, dtype=torch.int64)
  # Batched labels
  labels = torch.randint(0, num_classes, (8,), dtype=torch.int64)
  ```

- **Point clouds**: Dictionary format with mandatory 'pos' key
  ```python
  # Individual point cloud
  pc = {'pos': torch.randn(1024, 3), 'feat': torch.randn(1024, 32)}
  # Batched point clouds (concatenated along point dimension)
  pc = {'pos': torch.randn(2048, 3), 'feat': torch.randn(2048, 32)}
  ```

- **Model predictions**: Follow task-specific formats
  - Classification: `(N, C)` float32 for batched, `(C,)` for individual
  - Segmentation: `(N, C, H, W)` float32 for batched, `(C, H, W)` for individual
  - Depth: `(N, 1, H, W)` float32 for batched, `(1, H, W)` for individual
