# `models/three_d/point_cloud/render/` code skeleton

## Code implementation structure

`models/three_d/point_cloud/render/common/prepare_points_for_rendering.py`

```text
prepare_points_for_rendering.py
├── import math
├── from typing import Callable, Optional, Tuple
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.ops.world_to_camera_transform import world_to_camera_transform
├── def prepare_points_for_rendering(pc: PointCloud, camera: Camera, resolution: Tuple[int, int], max_divide: int = 0, num_divide: Optional[int] = None, cull_func: Callable[[torch.Tensor, torch.Tensor, int, int], None] = _frustum_cull) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Public entry that prepares the camera (opencv convention + resolution-scaled intrinsics) and adaptively batches point preprocessing to mitigate CUDA OOM.
│   ├── impls camera_prepared = camera.to(device=points.device, convention="opencv") then scale_intrinsics(resolution=resolution)
│   ├── if num_divide is not None
│   │   ├── impls batch_size = max(1, math.ceil(N / 2 ** num_divide))
│   │   ├── calls _prepare_points_for_rendering_batched  # camera=camera_prepared, batch_size fixed from num_divide
│   │   └── return  # the batched, depth-sorted result
│   ├── while n <= max_divide
│   │   ├── try
│   │   │   ├── calls _prepare_points_for_rendering_batched  # camera=camera_prepared, batch_size = ceil(N / 2 ** n)
│   │   │   └── return  # the batched, depth-sorted result
│   │   └── except torch.cuda.OutOfMemoryError
│   │       └── impls increment n to retry with a halved batch
│   └── raise  # torch.cuda.OutOfMemoryError once max_divide halvings are exhausted
├── def _prepare_points_for_rendering_batched(points: torch.Tensor, camera: Camera, resolution: Tuple[int, int], batch_size: int = 2048, cull_func: Callable[[torch.Tensor, torch.Tensor, int, int], None] = _frustum_cull) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Runs _prepare_points_for_rendering over fixed-size point batches, then concatenates and globally back-to-front depth-sorts the survivors.
│   ├── impls render_intrinsics = camera.intrinsics  # the CameraIntrinsics carries the camera-to-image projection
│   ├── impls extrinsics = camera.extrinsics.extrinsics  # the [4, 4] cam2world tensor
│   ├── for each batch [i:j] of points
│   │   └── calls _prepare_points_for_rendering  # render_intrinsics=render_intrinsics, extrinsics=extrinsics, cull_func=cull_func
│   ├── if no batch produced survivors
│   │   └── raise AssertionError  # no points remained after culling in all batches
│   ├── impls concatenate the per-batch survivors and their global indices  # impls-node-one-step:skip
│   └── impls globally depth-sort the concatenated points back-to-front by column 2
├── def _prepare_points_for_rendering(points: torch.Tensor, render_intrinsics: CameraIntrinsics, extrinsics: torch.Tensor, resolution: Tuple[int, int], cull_func: Callable[[torch.Tensor, torch.Tensor, int, int], None] = _frustum_cull) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Preprocesses one chunk of world-space points: world-to-camera transform, positive-depth filter, camera-to-image projection, then image-bounds cull.
│   ├── calls world_to_camera_transform  # points -> camera frame in place, forwarding extrinsics (world-to-camera step)
│   ├── impls keep only positive-depth points, compacting the surviving points/indices
│   ├── if nothing survives the depth filter
│   │   └── return  # empty points/indices for this batch
│   ├── calls render_intrinsics.project  # points_camera=current_points, inplace=True -> image (x, y) into columns 0, 1 (camera-to-image step)
│   ├── calls cull_func  # frustum cull against render_height / render_width, writing bounds_mask
│   ├── if nothing survives the bounds cull
│   │   └── return  # empty points/indices for this batch
│   └── return  # (points_2d [M, 3] as (x, y, depth), indices [M])
└── def _frustum_cull(current_points: torch.Tensor, bounds_mask: torch.Tensor, render_height: int, render_width: int) -> None
    ├── # Writes into bounds_mask whether each projected point lies within the image bounds (0 <= x < render_width, 0 <= y < render_height).
    └── impls set bounds_mask to the in-bounds test over current_points columns 0/1 against render_width / render_height
```
