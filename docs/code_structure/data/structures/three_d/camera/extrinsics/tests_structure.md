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
│   ├── # A [B, 3, 3] batch stabilized in one call matches stabilizing each rotation alone, and the cam2world batch it builds passes validate_camera_extrinsics for both float32 and float64.
│   ├── impls batch_size = 32
│   └── for dtype in (torch.float32, torch.float64)
│       ├── for each index below batch_size
│       │   ├── calls _random_rotation(dtype=dtype, seed=index)
│       │   ├── calls _random_rotation(dtype=dtype, seed=index + 5000)
│       │   └── impls multiply the two rotations into one near-orthogonal (3, 3) rotation
│       ├── impls rotations = those rotations stacked into a [batch_size, 3, 3] batch in dtype
│       ├── calls _stabilize_rotation_matrix(rotation=rotations)  # -> stabilized
│       ├── for each index, rotation of rotations
│       │   ├── calls _stabilize_rotation_matrix(rotation=rotation)  # -> stabilized_alone
│       │   └── assert stabilized[index] == stabilized_alone at every entry
│       ├── impls extrinsics = batch_size 4x4 identities in dtype, a [batch_size, 4, 4] stack
│       ├── impls extrinsics[:, :3, :3] = stabilized  # the (batch_size, 4, 4) cam2world batch
│       └── calls validate_camera_extrinsics(obj=extrinsics)
├── def test_stabilize_rejects_a_reflection
│   ├── # A batch mixing proper rotations with reflections is refused rather than sign-repaired, since a camera's rotation is proper by construction.
│   ├── impls batch_size = 4
│   ├── impls rotation_list = an empty list
│   ├── for each index below batch_size
│   │   ├── calls _random_rotation(dtype=torch.float64, seed=index)  # -> the proper rotation at that seed
│   │   └── impls append that proper rotation to rotation_list
│   ├── impls rotations = rotation_list stacked into a [batch_size, 3, 3] float64 batch  # the per-entry proper rotations
│   ├── impls rotations[1::2, :, 0] = -rotations[1::2, :, 0]  # every second entry column-negated into a reflection
│   └── with pytest.raises(AssertionError)
│       └── calls _stabilize_rotation_matrix(rotation=rotations)  # a [B, 3, 3] batch whose entries alternate a proper rotation and one column-negated into a reflection
├── def test_validator_threshold_is_dtype_aware
│   ├── # A fixed near-orthogonality deviation between the float64 and float32 tolerances passes validate_rotation_matrix as float32 but is rejected as float64.
│   ├── impls build a (3, 3) rotation whose orthogonality residual sits between the float64 and float32 tolerances  # impls-node-one-step:skip
│   ├── calls validate_rotation_matrix(obj=that rotation cast to float32)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls validate_rotation_matrix(obj=that rotation cast to float64)
│   └── return
├── def test_validator_requires_determinant_plus_one
│   ├── # A camera's rotation is validated to have determinant +1, so a reflection is inexpressible as camera extrinsics and any change of handedness has to be carried by the geometry instead.
│   ├── calls _random_rotation(dtype=torch.float64, seed=a fixed seed)
│   ├── calls validate_rotation_matrix(obj=that proper rotation of determinant +1)
│   ├── impls build an orthonormal (3, 3) matrix whose determinant is -1 by negating one of its columns
│   ├── with pytest.raises(AssertionError)
│   │   └── calls validate_rotation_matrix(obj=that determinant -1 matrix)
│   └── with pytest.raises(AssertionError)
│       └── calls validate_camera_extrinsics(obj=a cam2world batch carrying that reflection)
└── def _random_rotation(dtype: torch.dtype, seed: int) -> torch.Tensor
    ├── # One reproducible proper rotation per seed, so every rotation-building test starts from a real rotation instead of a hand-written matrix.
    ├── impls draw a seeded random (3, 3) float64 matrix
    ├── impls take the QR decomposition of that matrix
    ├── impls canonicalize the orthonormal factor by the signs of the triangular factor's diagonal
    ├── if that orthonormal factor has a negative determinant
    │   └── impls negate its first column
    ├── impls cast the orthonormal factor to the requested dtype
    └── return  # the orthonormal factor, a (3, 3) proper rotation in the requested dtype
```
