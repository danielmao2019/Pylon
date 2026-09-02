# Camera Extrinsics Tests Structure

## 1. Tests implementation structure

`tests/data/structures/three_d/camera/extrinsics/test_rotation_stabilize_validate_compat.py`

```text
test_rotation_stabilize_validate_compat.py
├── import pytest
├── import torch
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import _stabilize_rotation_matrix
├── from data.structures.three_d.camera.extrinsics.validation import validate_camera_extrinsics, validate_rotation_matrix
├── def test_stabilize_accepts_float32_and_float64
│   ├── # _stabilize_rotation_matrix accepts a float32 or float64 near-orthogonal rotation, returns the same dtype, and its output passes validate_rotation_matrix.
│   ├── for each dtype in {torch.float32, torch.float64}
│   │   ├── calls _stabilize_rotation_matrix(rotation=a near-orthogonal (3, 3) rotation in that dtype)
│   │   ├── impls assert the returned rotation keeps that dtype
│   │   └── calls validate_rotation_matrix(obj=the returned rotation)
│   └── return
├── def test_stabilize_rejects_unsupported_dtype
│   ├── # _stabilize_rotation_matrix raises on a dtype outside {float32, float64} (e.g. float16).
│   ├── with pytest.raises(AssertionError)
│   │   └── calls _stabilize_rotation_matrix(rotation=a float16 near-orthogonal rotation)
│   └── return
├── def test_stabilized_batch_passes_validator
│   ├── # A batch of stabilized cam2world extrinsics passes the batched validate_camera_extrinsics for both float32 and float64.
│   ├── for each dtype in {torch.float32, torch.float64}
│   │   ├── for each pose in the batch
│   │   │   └── calls _stabilize_rotation_matrix
│   │   ├── impls stack the stabilized rotations into a (B, 4, 4) cam2world batch in that dtype
│   │   └── calls validate_camera_extrinsics(obj=the (B, 4, 4) cam2world batch)
│   └── return
├── def test_validator_threshold_is_dtype_aware
│   ├── # A fixed near-orthogonality deviation between the float64 and float32 tolerances passes validate_rotation_matrix as float32 but is rejected as float64.
│   ├── impls build a (3, 3) rotation whose orthogonality residual sits between the float64 and float32 tolerances  # impls-node-one-step:skip
│   ├── calls validate_rotation_matrix(obj=that rotation cast to float32)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls validate_rotation_matrix(obj=that rotation cast to float64)
│   └── return
└── def test_validator_requires_determinant_plus_one
    ├── # A camera's rotation is validated to have determinant +1, so a reflection is inexpressible as camera extrinsics and any change of handedness has to be carried by the geometry instead.
    ├── calls validate_rotation_matrix(obj=a proper rotation of determinant +1)
    ├── impls build an orthonormal (3, 3) matrix whose determinant is -1 by negating one of its columns
    ├── with pytest.raises(AssertionError)
    │   └── calls validate_rotation_matrix(obj=that determinant -1 matrix)
    ├── with pytest.raises(AssertionError)
    │   └── calls validate_camera_extrinsics(obj=a cam2world batch carrying that reflection)
    └── return
```
