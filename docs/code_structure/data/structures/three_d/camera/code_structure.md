# Camera Data Structure Code Structure

## 1. Code structure trees

`./data/structures/three_d/camera/camera_vis.py`

```text
camera_vis.py
├── from typing import Any, Dict, List, Optional
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
├── def cameras_vis(cameras: Cameras, frustum_scale: float, frustum_color: Optional[torch.Tensor] = None) -> List[Dict[str, Any]]
│   ├── # Builds a camera-trajectory visualization payload from a Cameras collection.
│   ├── for each camera
│   │   └── calls camera_vis
│   └── return
└── def camera_vis(camera: Camera, frustum_scale: float, frustum_color: Optional[torch.Tensor] = None) -> Dict[str, Any]
    ├── # Builds one camera visualization primitive from a Camera whose intrinsics may be absent.
    ├── impls computes center, center_color, and axes from camera center, right, forward, and up
    ├── impls computes frustum lines from camera intrinsics and frustum_scale
    └── return
```
