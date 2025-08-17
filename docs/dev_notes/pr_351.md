# ResizeMaps Ignore Value Fix - Implementation Summary

## Problem Identified

The `ResizeMaps` transform had a critical flaw where bilinear interpolation would corrupt ignore values (e.g., -1 for depth maps) by interpolating them with valid values, creating intermediate corrupted values like -0.33, -0.71, etc.

### Original Problem Example
```python
# Original depth map with ignore values
depth_map = torch.ones((6, 6)) * 10.0  # Valid depth = 10.0  
depth_map[1::2, 1::2] = -1.0           # Ignore regions = -1.0

# Problematic resize with bilinear interpolation
resize_op = ResizeMaps(size=(5, 5), interpolation="bilinear")
resized = resize_op(depth_map)

# Result: corrupted values like -0.33, -0.71 instead of clean -1.0 or 10.0
print(resized.min())  # -0.71 (CORRUPTED!)
```

## Solution Implemented

### 1. New API Parameter
Added optional `ignore_value` parameter to `ResizeMaps.__init__()`:

```python
ResizeMaps(ignore_value=-1.0, size=(256, 256), interpolation="bilinear")
```

- **Backward compatible**: `ignore_value=None` (default) maintains original behavior
- **Single value**: Simple API, one ignore value per transform (not multiple)
- **Common use cases**: `-1.0` (depth), `255` (segmentation), `float('nan')` (missing data)

### 2. Mask-Based Interpolation Strategy with Tolerance Constant

When `ignore_value` is specified and bilinear interpolation is used:

1. **Create valid pixel mask with tolerance**: `valid_mask = torch.abs(x - ignore_value) >= ResizeMaps.TOLERANCE`
2. **Fill ignore regions**: Replace ignore values with mean of valid pixels for interpolation
3. **Resize data and mask separately**: Both use bilinear interpolation
4. **Restore ignore values**: Where `mask_resized <= 0.5`, set pixels back to `ignore_value`

**Key Improvements**: 
- Uses tolerance-based comparison (`abs(x - ignore_value) >= ResizeMaps.TOLERANCE`) instead of exact equality to handle floating-point precision issues correctly
- Defines tolerance as class constant `ResizeMaps.TOLERANCE = 1e-5` for consistency between implementation and tests

### 3. Conditional Logic

```python
if (self.ignore_value is not None and 
    resize_op.interpolation == torchvision.transforms.functional.InterpolationMode.BILINEAR):
    return self._ignore_aware_resize(x, resize_op)
else:
    return self._standard_resize(x, resize_op)
```

**Ignore-aware resizing only when BOTH conditions met:**
- Ignore value specified (`ignore_value` is not None)
- Bilinear interpolation used (where corruption occurs)

**Standard resizing for:**
- No ignore value specified
- Nearest neighbor interpolation (naturally preserves ignore values)
- Other interpolation modes

## Comprehensive Test Suite (`test_ignore_values.py`)

### Core Functionality Tests
- `test_resize_maps_ignore_values_bilinear()` - Primary corruption prevention test
- `test_resize_maps_ignore_values_nearest()` - Nearest neighbor verification  
- `test_resize_maps_depth_map_realistic()` - Real-world depth map scenario

### Edge Case Tests
- `test_resize_maps_ignore_values_segmentation_mask()` - Class 255 ignore handling
- `test_resize_maps_ignore_nan_values()` - NaN ignore values
- `test_resize_maps_no_ignore_value_specified()` - Backward compatibility
- `test_resize_maps_all_ignore_values()` - All pixels are ignore values
- `test_resize_maps_no_ignore_values_present()` - Optimization path testing

### Tolerance-Based Assertions with Class Constants
All tests use `ResizeMaps.TOLERANCE` for consistent and robust floating-point comparison:
```python
# Instead of exact equality
corrupted = (resized < 0) & (resized != ignore_value)  # ❌ Fragile

# Use tolerance-based comparison with class constant
corrupted = (resized < 0) & (torch.abs(resized - ignore_value) >= ResizeMaps.TOLERANCE)  # ✅ Robust

# Consistent tolerance checking across all tests
close_to_ignore = torch.abs(resized - ignore_value) < ResizeMaps.TOLERANCE
```

## Verification Results

**Before Fix:**
```
❌ Corrupted pixels: 3
❌ Intermediate values: [-0.33, -0.71] (interpolated ignore values)
❌ Boundary corruption: 4 pixels
```

**After Fix:**
```
✅ Corrupted pixels: 0  
✅ Only exact ignore values: [-1.0] or valid range [0.5, 10.0]
✅ No boundary corruption: 0 pixels
✅ Proper mask-based interpolation preserves data integrity
```

## Usage Examples

### For Depth Maps
```python
# Depth maps with -1.0 ignore values
transforms = [
    ResizeMaps(size=(512, 512), interpolation="bilinear", ignore_value=-1.0)
]
```

### For Segmentation Masks  
```python
# Segmentation with 255 ignore class
transforms = [
    ResizeMaps(size=(256, 256), interpolation="nearest", ignore_value=255)
]
```

### Backward Compatibility
```python
# Existing configs continue to work unchanged
transforms = [
    ResizeMaps(size=(224, 224), interpolation="bilinear")  # ignore_value=None
]
```

## Related Transforms That May Need Similar Fixes

**Identified transforms with potential interpolation issues:**
1. `Rotation` / `RandomRotation` - uses `torchvision.transforms.functional.rotate()`
2. `Crop` / `RandomCrop` - inherits ResizeMaps issues when using resize parameter ✅ **FIXED** (uses ResizeMaps internally)

**Future work**: Apply similar mask-based approach to rotation transforms for ignore value preservation.

## Impact

- **✅ Data Integrity**: Prevents silent corruption of ignore values during resizing
- **✅ Scientific Accuracy**: Depth maps and sensor data maintain measurement validity  
- **✅ Backward Compatibility**: Existing code continues to work unchanged
- **✅ Performance**: Mask-based approach adds minimal overhead only when needed
- **✅ Test Coverage**: Comprehensive test cases catch regression issues
- **✅ Code Consistency**: Tolerance constant ensures uniform behavior between implementation and tests

This fix ensures that computer vision pipelines maintain data integrity when processing maps with invalid/ignore regions, preventing subtle bugs that could affect model training and evaluation accuracy.