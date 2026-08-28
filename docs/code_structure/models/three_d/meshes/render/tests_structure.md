# `models/three_d/meshes/render/` tests skeleton

## Tests implementation structure

### tests/models/three_d/meshes/render/test_core.py

`tests/models/three_d/meshes/render/test_core.py`

```text
test_core.py
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from models.three_d.meshes.render.core import _prepare_cameras
├── def test_prepare_cameras_restates_both_camera_frames_for_pytorch3d() -> None
│   ├── # Both camera-frame changes belong to the Camera, so mesh rendering only hands PyTorch3D a camera already stated in PyTorch3D's frames.
│   ├── calls _build_camera
│   ├── calls _prepare_cameras(camera=camera, resolution=resolution, device=device)
│   ├── calls camera.to(device=device, intr_convention="pytorch3d", extr_convention="pytorch3d").scale_intrinsics(resolution=resolution)
│   ├── impls assert the prepared camera is in NDC
│   ├── impls assert focal_length matches the converted intrinsics
│   ├── impls assert principal_point matches the converted intrinsics
│   ├── impls assert R matches the converted world-to-camera pose
│   ├── impls assert T matches the converted world-to-camera pose
│   └── return
└── def _build_camera() -> Camera
    ├── # Builds a standard-pixel, standard-pose pinhole camera on the CPU.
    ├── calls build_camera_intrinsics
    ├── calls CameraExtrinsics
    ├── calls Camera
    └── return
```

### tests/models/three_d/meshes/render/test_shading.py

`tests/models/three_d/meshes/render/test_shading.py`

```text
test_shading.py
├── import pytest
├── from models.three_d.meshes.render.shading import compute_sh_shading
├── def test_band_count_selects_the_spherical_harmonic_order
│   ├── # Coefficients of any perfect-square band count evaluate at the order that count implies, so a caller's band count is never assumed.
│   ├── for each perfect-square band count
│   │   ├── calls compute_sh_shading
│   │   └── impls assert the shading has one RGB triple per input normal
│   └── return
├── def test_non_square_band_count_is_rejected
│   ├── # A coefficient set whose band count is not a perfect square names no spherical-harmonic order, so it fails the assertion rather than evaluating.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls compute_sh_shading
│   └── return
└── def test_shading_varies_with_the_normal_direction
    ├── # Two normals facing differently under the same non-constant coefficients receive different shading, so the basis is really evaluated over the normal.
    ├── calls compute_sh_shading
    ├── impls assert the two normals' shading differs
    └── return
```
