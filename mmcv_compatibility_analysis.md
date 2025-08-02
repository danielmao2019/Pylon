# MMCV 2.0.0 Compatibility Analysis for GMCNet

## Problem Summary

The original GMCNet implementation was built for mmcv 1.x but Pylon uses mmcv 2.0.0. Several API changes between versions are causing import failures:

## Compatibility Issues Found

### 1. Missing Modules in mmcv 2.0.0
- `mmcv.runner` → Does not exist in mmcv 2.0.0
- `mmcv.utils` → Does not exist in mmcv 2.0.0

### 2. Specific Import Failures
- `from mmcv.runner import force_fp32` → `mmcv.runner` module missing
- `from mmcv.utils import Registry` → `mmcv.utils` module missing

### 3. Available in mmcv 2.0.0
- `mmcv.cnn.ConvModule` ✅ Available
- Most core functionality is preserved

## Files Requiring Changes

1. **point_fp_module.py** - Uses `force_fp32` decorator
2. **builder.py** - Uses `Registry` from mmcv.utils
3. **norm.py** - Uses `force_fp32` and `NORM_LAYERS`
4. **points_sampler.py** - Uses `force_fp32`

## Proposed Solutions

### Option 1: Remove Decorators (Minimal Impact)
- Remove `@force_fp32()` decorators - they're optimization hints, not core functionality
- Replace `Registry` with simple dictionary-based registration
- Remove unused `NORM_LAYERS` imports

### Option 2: Create Compatibility Shims
- Create lightweight replacement decorators that do nothing
- Create minimal Registry class for backward compatibility

## Recommendation

**Option 1** is recommended because:
1. `force_fp32` is an optimization decorator that doesn't affect core functionality
2. The Registry usage appears minimal and can be simplified
3. This approach preserves the original logic while removing dependencies

## Files That Need Modification

1. `/models/point_cloud_registration/gmcnet/mm3d_pn2/ops/pointnet_modules/point_fp_module.py`
   - Remove `force_fp32` import and decorator
   
2. `/models/point_cloud_registration/gmcnet/mm3d_pn2/ops/pointnet_modules/builder.py`
   - Remove Registry dependency
   
3. `/models/point_cloud_registration/gmcnet/mm3d_pn2/ops/norm.py`
   - Remove `force_fp32` and `NORM_LAYERS` imports

4. `/models/point_cloud_registration/gmcnet/mm3d_pn2/ops/furthest_point_sample/points_sampler.py`
   - Remove `force_fp32` import and decorator

These are minimal changes that preserve functionality while fixing compatibility.