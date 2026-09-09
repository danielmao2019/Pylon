# `models/three_d/point_cloud/render/` code skeleton

## Code implementation structure

`models/three_d/point_cloud/render/common/create_circular_kernel_offsets.py`

```text
create_circular_kernel_offsets.py
├── import torch
└── def create_circular_kernel_offsets(point_size: float, device: torch.device) -> torch.Tensor
    ├── # Enumerates the (y, x) offsets of those cells of the kernel_size x kernel_size grid that lie within point_size / 2 of the origin.
    ├── impls kernel_size = int(torch.ceil(torch.tensor(point_size)))  # torch.tensor narrows a python float to float32 before the ceil
    ├── impls kernel_radius = point_size / 2.0
    ├── impls y_kernel, x_kernel = the ij-indexed meshgrid of two arange(kernel_size) axes on device, each shifted by subtracting kernel_size // 2
    ├── impls kernel_distances = the euclidean distance from the origin over the float-cast y_kernel, x_kernel grids
    ├── impls circular_mask = the elementwise kernel_distances <= kernel_radius boolean grid
    ├── impls kernel_offsets = y_kernel, x_kernel selected by circular_mask, stacked along dim 1
    └── return  # kernel_offsets [num_grid_cells_within_radius, 2] int64 on device as (y, x)
```

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
│   ├── impls camera_prepared = camera.to(device=points.device, dtype=points.dtype, extr_convention="opencv").scale_intrinsics(resolution=resolution)
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

`models/three_d/point_cloud/render/render_depth.py`

```text
render_depth.py
├── from typing import Tuple, Union
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.render.common.prepare_points_for_rendering import prepare_points_for_rendering
├── from models.three_d.point_cloud.render.common.validate_rendering_inputs import validate_rendering_inputs
├── from models.three_d.point_cloud.render.render_mask import render_mask_from_rendering_points
├── def render_depth_from_point_cloud(pc: PointCloud, camera: Camera, resolution: Tuple[int, int], ignore_value: float = -1.0, return_mask: bool = False, point_size: float = 1.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
│   ├── # Renders a point cloud through the camera to a depth map, chaining validation, projection, and rasterization.
│   ├── assert isinstance(pc, PointCloud)  # f"{type(pc)=}"
│   ├── calls validate_rendering_inputs(pc=pc, camera=camera, resolution=resolution, ignore_value=ignore_value, return_mask=return_mask, point_size=point_size)
│   ├── calls prepare_points_for_rendering(pc=pc, camera=camera, resolution=resolution)  # -> rendered_points, the first of the (points, indices) pair
│   ├── calls render_depth_from_rendering_points(rendering_points=rendered_points, resolution=resolution, ignore_value=ignore_value, return_mask=return_mask)
│   └── return  # the render_depth_from_rendering_points result, returned directly
└── def render_depth_from_rendering_points(rendering_points: torch.Tensor, resolution: Tuple[int, int], ignore_value: float = float('inf'), return_mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ├── # Rasterizes already-projected points into a depth map by writing each point's depth at its pixel.
    ├── impls render_height, render_width = resolution
    ├── impls depth_map = a [render_height, render_width] float32 tensor filled with ignore_value on the rendering_points device
    ├── impls assign rendering_points column 2 float-cast into depth_map by advanced indexing at rows from its long-cast column 1, cols from its long-cast column 0
    ├── if return_mask
    │   ├── calls render_mask_from_rendering_points(rendering_points=rendering_points, resolution=resolution, device=rendering_points.device)  # -> valid_mask
    │   └── return  # (depth_map, valid_mask)
    └── else
        └── return  # depth_map
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
