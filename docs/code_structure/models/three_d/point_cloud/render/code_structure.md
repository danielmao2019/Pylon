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
│   ├── # Public entry that prepares the camera (opencv extr_convention + resolution-scaled intrinsics) and adaptively batches point preprocessing to mitigate CUDA OOM.
│   ├── impls points = pc.xyz  # the [N, 3] world-space point tensor
│   ├── impls camera_prepared = camera.to(device=points.device, extr_convention="opencv").scale_intrinsics(resolution=resolution)
│   ├── impls N = points.shape[0]
│   ├── if num_divide is not None
│   │   ├── impls batch_size = max(1, math.ceil(N / 2 ** num_divide))
│   │   ├── calls _prepare_points_for_rendering_batched(points=points, camera=camera_prepared, batch_size=batch_size)
│   │   └── return  # the batched, depth-sorted result
│   ├── while n <= max_divide
│   │   ├── try
│   │   │   ├── calls _prepare_points_for_rendering_batched(points=points, camera=camera_prepared, batch_size=ceil(N / 2 ** n))
│   │   │   └── return  # the batched, depth-sorted result
│   │   └── except torch.cuda.OutOfMemoryError
│   │       └── impls increment n to retry with a halved batch
│   └── raise  # torch.cuda.OutOfMemoryError once max_divide halvings are exhausted
├── def _prepare_points_for_rendering_batched(points: torch.Tensor, camera: Camera, resolution: Tuple[int, int], batch_size: int = 2048, cull_func: Callable[[torch.Tensor, torch.Tensor, int, int], None] = _frustum_cull) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Runs _prepare_points_for_rendering over fixed-size point batches, then concatenates and globally back-to-front depth-sorts the survivors.
│   ├── impls render_intrinsics = camera.intrinsics      # the CameraIntrinsics carries the camera-to-image projection
│   ├── impls extrinsics = camera.extrinsics.extrinsics  # the [4, 4] cam2world tensor
│   ├── for each batch [i:j] of points
│   │   └── calls _prepare_points_for_rendering(render_intrinsics=render_intrinsics, extrinsics=extrinsics, cull_func=cull_func)
│   ├── if no batch produced survivors
│   │   └── raise AssertionError  # no points remained after culling in all batches
│   ├── impls concatenate the per-batch survivors and their global indices  # impls-node-one-step:skip
│   └── impls globally depth-sort the concatenated points back-to-front by column 2
├── def _prepare_points_for_rendering(points: torch.Tensor, render_intrinsics: CameraIntrinsics, extrinsics: torch.Tensor, resolution: Tuple[int, int], cull_func: Callable[[torch.Tensor, torch.Tensor, int, int], None] = _frustum_cull) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Preprocesses one chunk of world-space points: world-to-camera transform, positive-depth filter, camera-to-image projection, then image-bounds cull.
│   ├── calls world_to_camera_transform(points=points, extrinsics=extrinsics, inplace=True)  # the world-to-camera step
│   ├── impls keep only positive-depth points, compacting the surviving points/indices
│   ├── if nothing survives the depth filter
│   │   └── return  # empty points/indices for this batch
│   ├── calls render_intrinsics.project(points_camera=current_points, inplace=True)  # -> image (x, y) into columns 0, 1 (the camera-to-image step)
│   ├── calls cull_func(current_points=current_points, bounds_mask=bounds_mask, render_height=render_height, render_width=render_width)  # writes bounds_mask in place
│   ├── if nothing survives the bounds cull
│   │   └── return  # empty points/indices for this batch
│   └── return  # (points_2d [M, 3] as (x, y, depth), indices [M])
└── def _frustum_cull(current_points: torch.Tensor, bounds_mask: torch.Tensor, render_height: int, render_width: int) -> None
    ├── # Writes into bounds_mask whether each projected point lies within the image bounds (0 <= x < render_width, 0 <= y < render_height).
    └── impls set bounds_mask to the in-bounds test over current_points columns 0/1 against render_width / render_height
```

`models/three_d/point_cloud/render/render_rgb_volumetric.py`

```text
render_rgb_volumetric.py
├── import itertools
├── from typing import List
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
└── def gen_auxiliary_cameras(points: torch.Tensor, camera: Camera) -> List[Camera]
    ├── # Rings the primary view with offset cameras, so one input view still gives a volumetric fit a spread of poses to train against.
    ├── impls device = points.device
    ├── impls center = the mean of points over its point axis, as float32 on device
    ├── calls camera.to(device=device, extr_convention='standard')
    ├── impls extrinsics_standard = the extrinsics matrix of the camera it returned
    ├── impls camera_position = the translation column of extrinsics_standard
    ├── impls distance = the norm of camera_position minus center
    ├── assert distance is positive  # a camera sitting on the centre names no direction to step away along
    ├── impls step = half of distance
    ├── impls direction_specs = the normalized float32 vectors over itertools.product of minus one, zero and one taken three at a time, the all-zero one dropped  # impls-node-one-step:skip — one step; the "and" names what it is made of
    ├── impls auxiliary_cameras = an empty list
    ├── for each direction_unit in direction_specs
    │   ├── assert direction_unit is a 3-vector
    │   ├── impls position = camera_position stepped along direction_unit by step
    │   ├── impls aux_standard = a [4, 4] float32 block carrying the rotation of extrinsics_standard, position in its translation column, and one in its corner  # impls-node-one-step:skip — one step; the "and" names what it is made of
    │   ├── calls CameraExtrinsics(extrinsics=aux_standard, extr_convention='standard', device=device)
    │   ├── impls aux_extrinsics = the extrinsics it built
    │   ├── calls Camera(intrinsics=camera.intrinsics, extrinsics=aux_extrinsics, device=device)
    │   └── impls auxiliary_cameras gains that camera brought to the convention camera now carries, which the rebinding above left standard
    └── return auxiliary_cameras
```
