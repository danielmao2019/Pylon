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
Dictionary format with mandatory 'pos' key

```python
# Individual point cloud
pc = {'pos': torch.randn(1024, 3), 'feat': torch.randn(1024, 32)}
# Batched point clouds (concatenated along point dimension)
pc = {'pos': torch.randn(2048, 3), 'feat': torch.randn(2048, 32)}
```

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
