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
