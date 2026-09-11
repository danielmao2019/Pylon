# `models/three_d/point_cloud/render/` code skeleton

## Code implementation structure

`models/three_d/point_cloud/render/common/apply_point_size_postprocessing.py`

```text
apply_point_size_postprocessing.py
├── from typing import Union
├── import torch
├── from models.three_d.point_cloud.render.common.create_circular_kernel_offsets import create_circular_kernel_offsets
└── def apply_point_size_postprocessing(rendered_image: torch.Tensor, depth_map: torch.Tensor, point_size: float, ignore_value: Union[int, float]) -> torch.Tensor
    ├── # Dilates each rendered point into a disc of point_size pixels, letting a nearer point's value overwrite a farther one so the dilation respects the same occlusion the rasterizer resolved.
    ├── # The camera axes ride in front of the image axes, so one call dilates a whole batch and a single camera alike.
    ├── impls render_height, render_width = the last two axes of depth_map
    ├── impls channel_axis = whether rendered_image carries one more axis than depth_map, which is what distinguishes a [..., C, H, W] image from a [..., H, W] map
    ├── calls create_circular_kernel_offsets(point_size=point_size, device=rendered_image.device)
    ├── impls kernel_offsets = the [num_offsets, 2] (y, x) grid it returned
    ├── impls neighbor_depth = depth_map shifted by every kernel offset and stacked along a new offset axis, out-of-bounds shifts filled with positive infinity  # impls-node-one-step:skip
    ├── impls source_offset = the offset axis' argmin over neighbor_depth, naming for each pixel which shifted source is nearest
    ├── impls dilated_image = rendered_image gathered along the image axes at the shift source_offset names, broadcast across channel_axis  # impls-node-one-step:skip
    ├── impls dilated_image = ignore_value wherever the nearest neighbor_depth is still positive infinity, so a pixel no disc reached keeps the background  # impls-node-one-step:skip
    └── return dilated_image
```

`models/three_d/point_cloud/render/common/create_circular_kernel_offsets.py`

```text
create_circular_kernel_offsets.py
├── import math
├── import torch
└── def create_circular_kernel_offsets(point_size: float, device: torch.device) -> torch.Tensor
    ├── # Enumerates the (y, x) offsets of those grid cells whose centre lies within point_size / 2 of the origin.
    ├── impls kernel_radius = point_size / 2.0
    ├── impls axis_offsets = arange from minus the ceiling of kernel_radius through plus it, on device  # one arange about the origin, so the grid reaches equally on both sides and the disc is centred rather than lopsided at an even extent
    ├── impls y_kernel, x_kernel = the ij-indexed meshgrid of axis_offsets against itself
    ├── impls kernel_distances = the euclidean distance from the origin over the float-cast y_kernel, x_kernel grids
    ├── impls circular_mask = the elementwise kernel_distances <= kernel_radius boolean grid
    ├── impls kernel_offsets = y_kernel, x_kernel selected by circular_mask, stacked along dim 1
    └── return  # kernel_offsets [num_grid_cells_within_radius, 2] int64 on device as (y, x)
```

`models/three_d/point_cloud/render/common/prepare_points_for_rendering.py`

```text
prepare_points_for_rendering.py
├── import math
├── from typing import Callable, Optional, Tuple, Union
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.ops.world_to_camera_transform import world_to_camera_transform
├── def prepare_points_for_rendering(pc: PointCloud, camera: Union[Camera, Cameras], resolution: Tuple[int, int], max_divide: int = 0, num_divide: Optional[int] = None, cull_func: Callable[[torch.Tensor, torch.Tensor, int, int], None] = _frustum_cull) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Public entry that prepares the camera (opencv extr_convention + resolution-scaled intrinsics) and adaptively batches point preprocessing to mitigate CUDA OOM.
│   ├── # Row i of the returned points is point i of pc.xyz, so a per-point attribute is looked up by the same index a rasterizer resolves per pixel.
│   ├── impls points = pc.xyz  # the [N, 3] world-space point tensor
│   ├── impls camera_prepared = camera.to(device=points.device, extr_convention="opencv").scale_intrinsics(resolution=resolution)
│   ├── impls N = points.shape[0]
│   ├── if num_divide is not None
│   │   ├── impls chunk_size = max(1, math.ceil(N / 2 ** num_divide))
│   │   ├── calls _prepare_points_for_rendering_chunked(points=points, camera=camera_prepared, chunk_size=chunk_size)
│   │   └── return  # the chunked result
│   ├── while n <= max_divide
│   │   ├── try
│   │   │   ├── calls _prepare_points_for_rendering_chunked(points=points, camera=camera_prepared, chunk_size=ceil(N / 2 ** n))
│   │   │   └── return  # the chunked result
│   │   └── except torch.cuda.OutOfMemoryError
│   │       └── impls increment n to retry with a halved chunk
│   └── raise  # torch.cuda.OutOfMemoryError once max_divide halvings are exhausted
├── def _prepare_points_for_rendering_chunked(points: torch.Tensor, camera: Union[Camera, Cameras], resolution: Tuple[int, int], chunk_size: int = 2048, cull_func: Callable[[torch.Tensor, torch.Tensor, int, int], None] = _frustum_cull) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Runs _prepare_points_for_rendering over fixed-size point chunks and concatenates them, leaving the point axis in its input order so a row still names its own point.
│   ├── impls render_intrinsics = camera.intrinsics      # the CameraIntrinsics carries the camera-to-image projection
│   ├── impls extrinsics = camera.extrinsics.extrinsics  # the [..., 4, 4] cam2world matrix, one per camera the batch carries
│   ├── for each chunk [i:j] of points  # chunked over points for memory; the camera batch axis passes through whole
│   │   └── calls _prepare_points_for_rendering(render_intrinsics=render_intrinsics, extrinsics=extrinsics, cull_func=cull_func)
│   ├── if no point of any camera survived
│   │   └── raise AssertionError  # no points remained after culling in all chunks
│   ├── impls concatenate the per-chunk points and their validity along the point axis  # impls-node-one-step:skip
│   └── return  # (points_2d [..., N, 3], valid [..., N]), the point axis still in pc.xyz order
├── def _prepare_points_for_rendering(points: torch.Tensor, render_intrinsics: CameraIntrinsics, extrinsics: torch.Tensor, resolution: Tuple[int, int], cull_func: Callable[[torch.Tensor, torch.Tensor, int, int], None] = _frustum_cull) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Preprocesses one chunk of world-space points: world-to-camera transform, positive-depth filter, camera-to-image projection, then image-bounds cull, each survivor marked rather than compacted out.
│   ├── calls world_to_camera_transform(points=points, extrinsics=extrinsics)  # -> [..., n, 3], one camera frame per camera the extrinsics carries
│   ├── impls valid = the positive-depth test on column 2  # marked, not compacted: cameras cull different points, so compaction would leave each a different length
│   ├── calls render_intrinsics.project(points_camera=current_points, inplace=True)  # -> image (x, y) into columns 0, 1 (the camera-to-image step); the params broadcast against the point axis
│   ├── calls cull_func(current_points=current_points, bounds_mask=bounds_mask, render_height=render_height, render_width=render_width)  # writes bounds_mask in place
│   ├── impls valid = valid & bounds_mask
│   └── return  # (points_2d [..., n, 3] as (x, y, depth), valid [..., n])
└── def _frustum_cull(current_points: torch.Tensor, bounds_mask: torch.Tensor, render_height: int, render_width: int) -> None
    ├── # Writes into bounds_mask whether each projected point lies within the image bounds (0 <= x < render_width, 0 <= y < render_height).
    └── impls set bounds_mask to the in-bounds test over current_points columns 0/1 against render_width / render_height
```

`models/three_d/point_cloud/render/common/validate_rendering_inputs.py`

```text
validate_rendering_inputs.py
├── from typing import Optional, Tuple, Union
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
└── def validate_rendering_inputs(pc: PointCloud, camera: Union[Camera, Cameras], resolution: Tuple[int, int], ignore_value: Optional[Union[int, float]] = None, return_mask: bool = False, point_size: float = 1.0) -> None
    ├── # The precondition the depth, rgb, segmentation and normal entries assert before projecting: a point cloud sharing one device with its camera, a positive (height, width) pair, a point size of at least one pixel.
    ├── assert isinstance(pc, PointCloud)  # f"{type(pc)=}"
    ├── assert isinstance(camera, (Camera, Cameras))  # f"{type(camera)=}"; the checks below read only the two components, which a single camera and a batch both carry
    ├── impls points = pc.xyz
    ├── impls intrinsics = camera.intrinsics
    ├── impls extrinsics = camera.extrinsics
    ├── assert intrinsics.device == points.device             # f"points device {points.device} != camera_intrinsics device {intrinsics.device}"
    ├── assert extrinsics.device == points.device             # f"points device {points.device} != camera_extrinsics device {extrinsics.device}"
    ├── assert isinstance(resolution, (tuple, list))          # f"resolution must be tuple or list, got {type(resolution)}"
    ├── assert len(resolution) == 2                           # f"resolution must have 2 elements (height, width), got {len(resolution)}"
    ├── assert every element of resolution is a positive int  # f"resolution must be positive integers, got {resolution}"
    ├── if ignore_value is not None
    │   └── assert isinstance(ignore_value, (int, float))  # f"ignore_value must be int or float, got {type(ignore_value)}"
    ├── assert isinstance(return_mask, bool)         # f"return_mask must be bool, got {type(return_mask)}"
    ├── assert isinstance(point_size, (int, float))  # f"point_size must be numeric, got {type(point_size)}"
    └── assert point_size >= 1.0                     # f"point_size must be >= 1.0, got {point_size}"
```

`models/three_d/point_cloud/render/render_depth.py`

```text
render_depth.py
├── from typing import Optional, Tuple, Union
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.render.common.apply_point_size_postprocessing import apply_point_size_postprocessing
├── from models.three_d.point_cloud.render.common.prepare_points_for_rendering import prepare_points_for_rendering
├── from models.three_d.point_cloud.render.common.validate_rendering_inputs import validate_rendering_inputs
├── from models.three_d.point_cloud.render.render_mask import render_mask_from_rendering_points
├── def render_depth_from_point_cloud(pc: PointCloud, camera: Union[Camera, Cameras], resolution: Tuple[int, int], ignore_value: float = -1.0, return_mask: bool = False, point_size: float = 1.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
│   ├── # Renders a point cloud through the camera to a depth map, chaining validation, projection, and rasterization; a Camera gives [H, W] and a Cameras gives [B, H, W] down the same path.
│   ├── assert isinstance(pc, PointCloud)  # f"{type(pc)=}"
│   ├── calls validate_rendering_inputs(pc=pc, camera=camera, resolution=resolution, ignore_value=ignore_value, return_mask=return_mask, point_size=point_size)  # camera is whichever of Camera / Cameras the caller passed, so the shared preconditions are checked over either
│   ├── calls prepare_points_for_rendering(pc=pc, camera=camera, resolution=resolution)
│   ├── impls rendered_points, valid = the pair it returned
│   ├── if point_size > 1.0
│   │   ├── calls render_depth_from_rendering_points(rendering_points=rendered_points, resolution=resolution, ignore_value=float("inf"), return_mask=False, valid=valid)
│   │   ├── impls depth_map = the map it returned, positive infinity wherever no point landed
│   │   ├── calls apply_point_size_postprocessing(rendered_image=depth_map, depth_map=depth_map, point_size=point_size, ignore_value=float("inf"))
│   │   ├── impls depth_map = the dilated map it returned
│   │   ├── impls covered = the finite pixels of depth_map  # the discs the dilation reached, read off the infinity sentinel rather than ignore_value, which may be NaN
│   │   └── impls depth_map = depth_map with ignore_value written wherever covered is False
│   ├── else
│   │   ├── calls render_depth_from_rendering_points(rendering_points=rendered_points, resolution=resolution, ignore_value=ignore_value, return_mask=False, valid=valid)
│   │   └── impls depth_map = the map it returned
│   ├── if return_mask
│   │   ├── if point_size > 1.0
│   │   │   └── impls valid_mask = covered  # the coverage the map's own dilation reached, so the mask and the map it describes cannot drift apart
│   │   ├── else
│   │   │   ├── calls render_mask_from_rendering_points(rendering_points=rendered_points, resolution=resolution, device=rendered_points.device, valid=valid)
│   │   │   └── impls valid_mask = the mask it rasterized
│   │   └── return  # (depth_map, valid_mask)
│   └── else
│       └── return  # depth_map
└── def render_depth_from_rendering_points(rendering_points: torch.Tensor, resolution: Tuple[int, int], ignore_value: float = float('inf'), return_mask: bool = False, valid: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ├── # Rasterizes already-projected points into a depth map by reading the depth of the point that owns each pixel.
    ├── impls winner = a [..., render_height, render_width] tensor holding, per pixel, the index along the point axis of the valid point with the smallest depth landing there, and -1 where none landed  # reduced per pixel rather than scattered, so occlusion does not depend on which write lands last
    ├── impls depth_map = column 2 of rendering_points gathered at winner, float32, with ignore_value wherever winner is -1  # impls-node-one-step:skip
    ├── if return_mask
    │   ├── calls render_mask_from_rendering_points(rendering_points=rendering_points, resolution=resolution, device=rendering_points.device, valid=valid)  # -> valid_mask
    │   └── return  # (depth_map, valid_mask)
    └── else
        └── return  # depth_map
```

`models/three_d/point_cloud/render/render_mask.py`

```text
render_mask.py
├── from typing import Optional, Tuple
├── import torch
└── def render_mask_from_rendering_points(rendering_points: torch.Tensor, resolution: Tuple[int, int], device: torch.device, valid: Optional[torch.Tensor] = None) -> torch.Tensor
    ├── # Marks the pixels a surviving point landed on, which is what distinguishes a rendered image's covered pixels from its background.
    ├── impls render_height, render_width = resolution
    ├── if valid is None
    │   └── impls valid = an all-True [..., N] bool tensor over the point axis of rendering_points
    ├── impls nearest_point_index = a [..., render_height, render_width] tensor holding, per pixel, the index along the point axis of the valid point with the smallest depth landing there, and -1 where none landed  # reduced per pixel rather than scattered, so occlusion does not depend on which write lands last
    ├── impls valid_mask = nearest_point_index >= 0, on device, carrying the leading axes of rendering_points
    └── return valid_mask
```

`models/three_d/point_cloud/render/render_normal.py`

```text
render_normal.py
├── from typing import Tuple, Union
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.render.common.apply_point_size_postprocessing import apply_point_size_postprocessing
├── from models.three_d.point_cloud.render.common.prepare_points_for_rendering import prepare_points_for_rendering
├── from models.three_d.point_cloud.render.common.validate_rendering_inputs import validate_rendering_inputs
├── from models.three_d.point_cloud.render.render_depth import render_depth_from_rendering_points
├── from models.three_d.point_cloud.render.render_mask import render_mask_from_rendering_points
├── def render_normal_from_point_cloud_3d(pc: PointCloud, camera: Camera, resolution: Tuple[int, int], ignore_value: float = 0.0, return_mask: bool = False, point_size: float = 1.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
│   ├── # Renders the normals a point cloud already carries through the camera to a normal map, chaining validation, projection, rasterization, and point-size dilation.
│   ├── assert isinstance(pc, PointCloud)  # f"{type(pc)=}"
│   ├── assert hasattr(pc, "normals")      # "PointCloud must contain normals field"
│   ├── calls validate_rendering_inputs(pc=pc, camera=camera, resolution=resolution, ignore_value=ignore_value, return_mask=return_mask, point_size=point_size)
│   ├── calls prepare_points_for_rendering(pc=pc, camera=camera, resolution=resolution)
│   ├── impls rendering_points, valid = the pair it returned
│   ├── calls render_normal_from_rendering_points_3d(rendering_points=rendering_points, valid=valid, pc_data=pc, camera=camera, resolution=resolution, ignore_value=ignore_value)
│   ├── impls normal_map = the map it rasterized
│   ├── if point_size > 1.0
│   │   ├── calls render_depth_from_rendering_points(rendering_points=rendering_points, resolution=resolution, ignore_value=float("inf"), return_mask=False, valid=valid)
│   │   ├── impls depth_map = the depth map it rasterized
│   │   ├── calls apply_point_size_postprocessing(rendered_image=depth_map, depth_map=depth_map, point_size=point_size, ignore_value=float("inf"))
│   │   ├── impls covered = the finite pixels of the dilated depth map  # the disc each surviving point reached, which is both this renderer's mask and its background
│   │   ├── calls apply_point_size_postprocessing(rendered_image=normal_map, depth_map=depth_map, point_size=point_size, ignore_value=float("inf"))
│   │   ├── impls normal_map = the dilated map it returned
│   │   ├── impls normal_map = normal_map with ignore_value written back wherever covered is False  # the helper fills with the depth sentinel it was handed, which is not this renderer's own background
│   │   ├── impls valid_pixels = the pixels where any normal_map channel differs from ignore_value
│   │   └── impls normal_map at valid_pixels = those columns unit-normalized over the channel dim
│   ├── if return_mask
│   │   ├── if point_size > 1.0
│   │   │   └── impls valid_mask = covered  # the coverage the image's own dilation reached, so the mask and the image it describes cannot drift apart
│   │   ├── else
│   │   │   ├── calls render_mask_from_rendering_points(rendering_points=rendering_points, resolution=resolution, device=rendering_points.device, valid=valid)
│   │   │   └── impls valid_mask = the mask it rasterized
│   │   └── return  # (normal_map, valid_mask)
│   └── else
│       └── return  # normal_map
└── def render_normal_from_rendering_points_3d(rendering_points: torch.Tensor, valid: torch.Tensor, pc_data: PointCloud, camera: Camera, resolution: Tuple[int, int], ignore_value: float = 0.0) -> torch.Tensor
    ├── # Rasterizes already-projected points into a normal map, rotating each visible point's world normal into the opencv camera frame on the way.
    ├── impls render_height, render_width = resolution
    ├── impls world_normals = pc_data.normals
    ├── assert world_normals.shape[0] == pc_data.xyz.shape[0]  # f"Normals count {world_normals.shape[0]} must match points count {pc_data.xyz.shape[0]}"
    ├── assert world_normals.shape[1] == 3                     # f"Normals must be 3D vectors, got shape {world_normals.shape}"
    ├── impls world_normals = world_normals unit-normalized over its last dim
    ├── impls winner = a [render_height, render_width] tensor holding, per pixel, the index along the point axis of the valid point with the smallest depth landing there, and -1 where none landed  # reduced per pixel rather than scattered, so occlusion does not depend on which write lands last
    ├── impls visible_world_normals = world_normals gathered at winner
    ├── calls camera.to(device=rendering_points.device, extr_convention="opencv")
    ├── impls camera = the opencv-convention copy on the rendering_points device it returned
    ├── impls rotation_matrix = the top-left 3x3 block of camera's w2c matrix
    ├── impls camera_normals = visible_world_normals right-multiplied by the transposed rotation_matrix
    ├── impls camera_normals = camera_normals unit-normalized over its last dim
    ├── impls normal_map = camera_normals moved channel-first, with ignore_value wherever winner is -1  # impls-node-one-step:skip
    └── return normal_map
```

`models/three_d/point_cloud/render/render_rgb.py`

```text
render_rgb.py
├── from typing import Tuple, Union
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.render.common.apply_point_size_postprocessing import apply_point_size_postprocessing
├── from models.three_d.point_cloud.render.common.prepare_points_for_rendering import prepare_points_for_rendering
├── from models.three_d.point_cloud.render.common.validate_rendering_inputs import validate_rendering_inputs
├── from models.three_d.point_cloud.render.render_depth import render_depth_from_rendering_points
├── from models.three_d.point_cloud.render.render_mask import render_mask_from_rendering_points
├── def render_rgb_from_point_cloud(pc: PointCloud, camera: Camera, resolution: Tuple[int, int], ignore_value: float = 0.0, return_mask: bool = False, point_size: float = 1.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
│   ├── # Renders a point cloud's colours through the camera to an RGB image, chaining validation, projection, rasterization, and point-size dilation.
│   ├── assert isinstance(pc, PointCloud)  # f"{type(pc)=}"
│   ├── assert hasattr(pc, "rgb")          # "PointCloud must contain rgb field"
│   ├── calls validate_rendering_inputs(pc=pc, camera=camera, resolution=resolution, ignore_value=ignore_value, return_mask=return_mask, point_size=point_size)
│   ├── calls prepare_points_for_rendering(pc=pc, camera=camera, resolution=resolution)
│   ├── impls rendering_points, valid = the pair it returned
│   ├── calls render_rgb_from_rendering_points(rendering_points=rendering_points, valid=valid, pc=pc, resolution=resolution, ignore_value=ignore_value)
│   ├── impls rgb_image = the image it rasterized
│   ├── if point_size > 1.0
│   │   ├── calls render_depth_from_rendering_points(rendering_points=rendering_points, resolution=resolution, ignore_value=float("inf"), return_mask=False, valid=valid)
│   │   ├── impls depth_map = the depth map it rasterized
│   │   ├── calls apply_point_size_postprocessing(rendered_image=depth_map, depth_map=depth_map, point_size=point_size, ignore_value=float("inf"))
│   │   ├── impls covered = the finite pixels of the dilated depth map  # the disc each surviving point reached, which is both this renderer's mask and its background
│   │   ├── calls apply_point_size_postprocessing(rendered_image=rgb_image, depth_map=depth_map, point_size=point_size, ignore_value=float("inf"))
│   │   ├── impls rgb_image = the dilated image it returned
│   │   └── impls rgb_image = rgb_image with ignore_value written back wherever covered is False  # the helper fills with the depth sentinel it was handed, which is not this renderer's own background
│   ├── if return_mask
│   │   ├── if point_size > 1.0
│   │   │   └── impls valid_mask = covered  # the coverage the image's own dilation reached, so the mask and the image it describes cannot drift apart
│   │   ├── else
│   │   │   ├── calls render_mask_from_rendering_points(rendering_points=rendering_points, resolution=resolution, device=rendering_points.device, valid=valid)
│   │   │   └── impls valid_mask = the mask it rasterized
│   │   └── return  # (rgb_image, valid_mask)
│   └── else
│       └── return  # rgb_image
└── def render_rgb_from_rendering_points(rendering_points: torch.Tensor, valid: torch.Tensor, pc: PointCloud, resolution: Tuple[int, int], ignore_value: float = 0.0) -> torch.Tensor
    ├── # Rasterizes already-projected points into an RGB image by writing, at each pixel, the colour of the point that owns it.
    ├── assert hasattr(pc, "rgb")  # "PointCloud missing rgb field"
    ├── impls render_height, render_width = resolution
    ├── impls colors = pc.rgb
    ├── assert colors.numel() > 0  # f"Colors tensor must not be empty, got {colors.numel()} elements"
    ├── impls colors = a clone of colors
    ├── impls integer_dtypes = the torch integer dtypes uint8 through int64
    ├── impls is_integer_dtype = whether the colors dtype is one of integer_dtypes
    ├── impls is_in_255_range = whether colors sits within [0, 255] with a maximum above 1.0
    ├── if is_integer_dtype or is_in_255_range
    │   └── impls colors = colors divided by 255.0
    ├── impls colors = colors clamped to [0.0, 1.0]
    ├── impls winner = a [render_height, render_width] tensor holding, per pixel, the index along the point axis of the valid point with the smallest depth landing there, and -1 where none landed  # reduced per pixel rather than scattered, so occlusion does not depend on which write lands last
    ├── impls pixel_colors = colors gathered at winner, channel-first
    ├── impls rgb_image = pixel_colors float-cast, with ignore_value wherever winner is -1  # impls-node-one-step:skip
    └── return rgb_image
```

`models/three_d/point_cloud/render/render_rgb_volumetric.py`

```text
render_rgb_volumetric.py
├── import itertools
├── import logging
├── import math
├── import subprocess
├── import tempfile
├── import time
├── from pathlib import Path
├── from typing import Any, Dict, List, Tuple
├── import numpy as np
├── import torch
├── from PIL import Image
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.nerfstudio.nerfstudio_data import NerfStudio_Data
├── from data.structures.three_d.point_cloud.io.save_point_cloud import save_point_cloud
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.structures.three_d.point_cloud.select import Select
├── from models.three_d.point_cloud.render.common.prepare_points_for_rendering import prepare_points_for_rendering
├── from models.three_d.point_cloud.render.render_rgb import render_rgb_from_point_cloud
├── from models.three_d.splatfacto.load_splatfacto import load_splatfacto_model
├── from models.three_d.splatfacto.render import render_rgb_from_splatfacto
├── def render_rgb_from_point_cloud_volumetric(pc: PointCloud, camera: Camera, resolution: Tuple[int, int], debug: bool = False) -> torch.Tensor
│   ├── # Renders one view volumetrically: cull to the points that project, ring the view with auxiliary cameras, train a splatfacto model on that tiny dataset, evaluate it back at the original camera.
│   ├── impls total_start = time.time()
│   ├── impls log the pipeline start
│   ├── assert isinstance(pc, PointCloud)  # f"{type(pc)=}"
│   ├── assert isinstance(camera, Camera)  # f"{type(camera)=}"
│   ├── impls render_height, render_width = resolution
│   ├── assert both render dimensions are positive  # "Render resolution must be positive"
│   ├── impls intrinsics = camera.intrinsics
│   ├── impls extrinsics = camera.extrinsics
│   ├── impls convention = camera.extrinsics.extr_convention
│   ├── impls native_width = twice intrinsics.cx, rounded to an int
│   ├── impls native_height = twice intrinsics.cy, rounded to an int
│   ├── assert both native dimensions are positive  # "Invalid image dimensions inferred from intrinsics"
│   ├── impls downscale_ratio_w = native_width / render_width
│   ├── impls downscale_ratio_h = native_height / render_height
│   ├── impls downscale_estimate = the mean of the two ratios
│   ├── impls valid_factors = [1, 2, 4, 8]
│   ├── impls downscale_factor = the valid factor nearest downscale_estimate
│   ├── assert math.isfinite(downscale_estimate) with both ratios within 0.01 of downscale_factor  # "Render resolution does not correspond to a supported downscale factor"
│   ├── impls stage_start = time.time()
│   ├── calls prepare_points_for_rendering(pc=pc, camera=camera, resolution=resolution)
│   ├── impls image_plane_points_indices = the point indices where the pair's validity mask is True
│   ├── calls Select(indices=image_plane_points_indices)
│   ├── impls pc = that selector applied to pc, keeping only the points that projected into the image
│   ├── calls gen_auxiliary_cameras(points=pc.xyz, camera=camera)
│   ├── impls aux_cameras = the shell of offset cameras it built
│   ├── impls train_extrinsics = the primary extrinsics followed by each auxiliary camera's extrinsics
│   ├── impls log the culling stage duration with the training-camera count
│   ├── impls stage_start = time.time()
│   ├── impls images, masks = two empty lists
│   ├── for each _extrinsics in train_extrinsics
│   │   ├── calls Camera(intrinsics=intrinsics, extrinsics=_extrinsics, device=pc.device)
│   │   ├── impls render_camera = the camera it built
│   │   ├── calls render_rgb_from_point_cloud(pc=pc, camera=render_camera, resolution=resolution, return_mask=True)
│   │   └── impls images, masks each gain the image, mask pair it returned
│   ├── impls log the base-render stage duration with the image count
│   ├── impls target_device = pc.xyz.device
│   ├── if debug
│   │   ├── impls tempdir = ./test_volumetric_rendering, created with its parents
│   │   ├── impls cleanup_fn = None
│   │   └── impls log the retained workspace path
│   ├── else
│   │   ├── impls temp_dir_context = a tempfile.TemporaryDirectory()
│   │   ├── impls tempdir = the context's name as a Path
│   │   └── impls cleanup_fn = the context's bound cleanup
│   ├── try
│   │   ├── impls stage_start = time.time()
│   │   ├── impls log the tempdir the dataset is written to
│   │   ├── calls _create_images(images=images, output_root=tempdir, downscale_factor=downscale_factor)
│   │   ├── calls _create_masks(masks=masks, output_root=tempdir, downscale_factor=downscale_factor)
│   │   ├── calls _create_ply(pc=pc, output_root=tempdir)
│   │   ├── calls _create_nerfstudio(intrinsics=intrinsics, train_extrinsics=train_extrinsics, eval_extrinsics=extrinsics, convention=convention, output_root=tempdir)
│   │   ├── impls log the dataset-write stage duration
│   │   ├── impls dataset_root = Path(tempdir)
│   │   ├── impls stage_start = time.time()
│   │   ├── calls _run_ns_train_splatfacto(dataset_root=dataset_root, downscale_factor=downscale_factor)
│   │   ├── impls model_dir = the run directory it returned
│   │   ├── impls log the ns-train stage duration
│   │   ├── impls stage_start = time.time()
│   │   ├── calls _assert_checkpoint_exists(model_dir=model_dir)
│   │   ├── calls load_splatfacto_model(model_dir=str(model_dir), device=target_device)
│   │   ├── impls pipeline = the model it loaded
│   │   ├── impls log the model-load stage duration
│   │   ├── impls stage_start = time.time()
│   │   ├── calls render_rgb_from_splatfacto(model=pipeline, camera=camera, resolution=resolution)
│   │   ├── impls rendered_image = the image it rendered
│   │   └── impls log the evaluation-render stage duration
│   ├── finally
│   │   └── if cleanup_fn is not None
│   │       └── impls invoke cleanup_fn to drop the temporary workspace
│   ├── impls log the total pipeline duration
│   ├── impls rendered_image = rendered_image moved onto target_device
│   └── return rendered_image
├── def gen_auxiliary_cameras(points: torch.Tensor, camera: Camera) -> List[Camera]
│   ├── # Rings the primary view with offset cameras, so one input view still gives a volumetric fit a spread of poses to train against.
│   ├── impls device = points.device
│   ├── impls center = the mean of points over its point axis, as float32 on device
│   ├── calls camera.to(device=device, extr_convention='standard')
│   ├── impls extrinsics_standard = the extrinsics matrix of the camera it returned
│   ├── impls camera_position = the translation column of extrinsics_standard
│   ├── impls distance = the norm of camera_position minus center
│   ├── assert distance is positive  # a camera sitting on the centre names no direction to step away along
│   ├── impls step = half of distance
│   ├── impls direction_specs = the normalized float32 vectors over itertools.product of minus one, zero and one taken three at a time, the all-zero one dropped  # impls-node-one-step:skip — one step; the "and" names what it is made of
│   ├── impls auxiliary_cameras = an empty list
│   ├── for each direction_unit in direction_specs
│   │   ├── assert direction_unit is a 3-vector
│   │   ├── impls position = camera_position stepped along direction_unit by step
│   │   ├── impls aux_standard = a [4, 4] float32 block carrying the rotation of extrinsics_standard, position in its translation column, and one in its corner  # impls-node-one-step:skip — one step; the "and" names what it is made of
│   │   ├── calls CameraExtrinsics(extrinsics=aux_standard, extr_convention='standard', device=device)
│   │   ├── impls aux_extrinsics = the extrinsics it built
│   │   ├── calls Camera(intrinsics=camera.intrinsics, extrinsics=aux_extrinsics, device=device)
│   │   └── impls auxiliary_cameras gains that camera brought to the convention camera now carries, which the rebinding above left standard
│   └── return auxiliary_cameras
├── def _create_images(images: List[torch.Tensor], output_root: str, downscale_factor: int) -> None
│   ├── # Writes the rendered RGB tensors out as the downscale-suffixed images directory a nerfstudio dataset reads.
│   ├── impls root = Path(output_root)
│   ├── impls suffix = the underscored downscale_factor when it exceeds one, else the empty string
│   ├── impls image_dir = root / f"images{suffix}"
│   ├── impls create image_dir with its parents
│   └── for each idx, image in enumerate(images)
│       ├── impls tensor = image detached onto cpu, clamped to [0.0, 1.0], as float32
│       ├── impls array = tensor permuted to HWC, scaled by 255.0, rounded, cast to np.uint8
│       ├── impls file_path = image_dir / f"image_{idx:02d}.png"
│       └── impls write array to file_path as a PIL Image
├── def _create_masks(masks: List[torch.Tensor], output_root: str, downscale_factor: int) -> None
│   ├── # Writes the rendered coverage masks out as the downscale-suffixed masks directory a nerfstudio dataset reads.
│   ├── impls root = Path(output_root)
│   ├── impls suffix = the underscored downscale_factor when it exceeds one, else the empty string
│   ├── impls mask_dir = root / f"masks{suffix}"
│   ├── impls create mask_dir with its parents
│   └── for each idx, mask in enumerate(masks)
│       ├── impls tensor = mask detached onto cpu, as bool
│       ├── impls array = tensor as np.uint8 scaled by 255
│       ├── impls file_path = mask_dir / f"mask_{idx:02d}.png"
│       └── impls write array to file_path as a mode-"L" PIL Image
├── def _create_ply(pc: PointCloud, output_root: str) -> None
│   ├── # Writes the culled point cloud as the point_cloud.ply the nerfstudio dataset seeds its gaussians from.
│   ├── impls root = Path(output_root)
│   ├── impls ply_path = root / "point_cloud.ply"
│   ├── assert isinstance(pc, PointCloud)  # f"{type(pc)=}"
│   └── calls save_point_cloud(pc, str(ply_path))
├── def _create_nerfstudio(cameras: List[Camera], output_root: Path) -> None
│   ├── # Writes the transforms.json a nerfstudio dataset is read through, carrying the shared intrinsics beside every training pose.
│   ├── impls root = Path(output_root)
│   ├── assert cameras is non-empty  # "At least one camera required to write transforms.json"
│   ├── impls nerfstudio_path = root / "transforms.json"
│   ├── impls create the parent directory of nerfstudio_path
│   ├── impls camera_names = the name of each camera
│   ├── assert no entry of camera_names is None  # f"{camera_names=}"
│   ├── impls camera_intrinsics = the intrinsics of the first camera
│   ├── impls intrinsic_params = a dict of fl_x, fl_y, cx, cy off camera_intrinsics, its four distortion terms zeroed
│   ├── impls resolution = twice camera_intrinsics.cy by twice camera_intrinsics.cx, each rounded to an int
│   ├── impls camera_model = "OPENCV"
│   ├── impls intrinsics = the [3, 3] float32 pinhole matrix of camera_intrinsics on the first camera's device
│   ├── impls applied_transform = the [3, 4] float32 array sending (x, y, z) to (x, z, -y)
│   ├── calls build_camera_intrinsics(model=camera_intrinsics.model, params=each of camera_intrinsics' params stacked over cameras, intr_convention=camera_intrinsics.intr_convention)
│   ├── impls batched_intrinsics = the intrinsics it built, one entry per camera along its leading axis
│   ├── calls CameraExtrinsics(extrinsics=every camera's extrinsics matrix stacked to [B, 4, 4], extr_convention=the first camera's extr_convention)
│   ├── impls batched_extrinsics = the [B, 4, 4] extrinsics it built
│   ├── calls Cameras(intrinsics=batched_intrinsics, extrinsics=batched_extrinsics, names=camera_names, ids=[camera.id for camera in cameras], device=cameras[0].device)
│   ├── impls nerfstudio_cameras = the Cameras it built
│   ├── impls modalities = ["image"]
│   ├── impls payload = an empty Dict[str, Any]
│   ├── calls NerfStudio_Data(data=payload, device=cameras[0].device, intrinsic_params=intrinsic_params, resolution=resolution, camera_model=camera_model, intrinsics=intrinsics, applied_transform=applied_transform, ply_file_path="point_cloud.ply", cameras=nerfstudio_cameras, modalities=modalities, train_filenames=None, val_filenames=None, test_filenames=None)
│   ├── impls nerfstudio_data = the NerfStudio_Data it built
│   └── calls nerfstudio_data.save(output_path=nerfstudio_path)
├── def _run_ns_train_splatfacto(dataset_root: Path, downscale_factor: int) -> Path
│   ├── # Trains a splatfacto model on the written dataset by shelling out to nerfstudio's ns-train, handing back the run directory it produced.
│   ├── impls output_dir = dataset_root / "outputs"
│   ├── impls create output_dir with its parents
│   ├── impls ns_train_cmd = the ns-train splatfacto argv over output_dir, dataset_root, the downscale factor, with sharing off, quit-on-train-completion on, a nerfstudio-data fraction eval mode at a 1.0 train split
│   ├── impls run ns_train_cmd through subprocess with check=True
│   ├── impls config_paths = every config.yml under output_dir, newest mtime first
│   ├── assert config_paths is non-empty  # f"ns-train did not create any configs under {output_dir}"
│   ├── impls model_dir = the parent directory of the newest entry of config_paths
│   └── return model_dir
└── def _assert_checkpoint_exists(model_dir: Path) -> Path
    ├── # Refuses a run that stopped short of the 30K-iteration checkpoint the volumetric render loads.
    ├── impls checkpoint_path = model_dir / "nerfstudio_models" / f"step-000029999.ckpt"
    ├── assert checkpoint_path.is_file()  # f"Training did not reach 30K iterations; missing checkpoint {checkpoint_path}"
    └── return checkpoint_path
```

`models/three_d/point_cloud/render/render_segmentation.py`

```text
render_segmentation.py
├── from typing import Tuple, Union
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.render.common.apply_point_size_postprocessing import apply_point_size_postprocessing
├── from models.three_d.point_cloud.render.common.prepare_points_for_rendering import prepare_points_for_rendering
├── from models.three_d.point_cloud.render.common.validate_rendering_inputs import validate_rendering_inputs
├── from models.three_d.point_cloud.render.render_depth import render_depth_from_rendering_points
├── from models.three_d.point_cloud.render.render_mask import render_mask_from_rendering_points
├── def render_segmentation_from_point_cloud(pc: PointCloud, key: str, camera: Camera, resolution: Tuple[int, int], ignore_value: int = 255, return_mask: bool = False, point_size: float = 1.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
│   ├── # Renders the labels a point cloud carries under key through the camera to a segmentation map, chaining validation, projection, rasterization, and point-size dilation.
│   ├── assert isinstance(pc, PointCloud)  # f"{type(pc)=}"
│   ├── assert hasattr(pc, key)            # f"PointCloud must contain '{key}' field"
│   ├── calls validate_rendering_inputs(pc=pc, camera=camera, resolution=resolution, ignore_value=ignore_value, return_mask=return_mask, point_size=point_size)
│   ├── calls prepare_points_for_rendering(pc=pc, camera=camera, resolution=resolution)
│   ├── impls rendering_points, valid = the pair it returned
│   ├── calls render_segmentation_from_rendering_points(rendering_points=rendering_points, valid=valid, pc=pc, key=key, resolution=resolution, ignore_value=ignore_value)
│   ├── impls seg_map = the map it rasterized
│   ├── if point_size > 1.0
│   │   ├── calls render_depth_from_rendering_points(rendering_points=rendering_points, valid=valid, resolution=resolution, ignore_value=float("inf"), return_mask=False)
│   │   ├── impls depth_map = the depth map it rasterized
│   │   ├── calls apply_point_size_postprocessing(rendered_image=depth_map, depth_map=depth_map, point_size=point_size, ignore_value=float("inf"))
│   │   ├── impls covered = the finite pixels of the dilated depth map  # the disc each surviving point reached, which is both this renderer's mask and its background
│   │   ├── calls apply_point_size_postprocessing(rendered_image=seg_map.float(), depth_map=depth_map, point_size=point_size, ignore_value=float("inf"))
│   │   ├── impls seg_map = the long cast of the dilated map it returned
│   │   └── impls seg_map = seg_map with ignore_value written back wherever covered is False  # the helper fills with the depth sentinel it was handed, which is not this renderer's own background
│   ├── if return_mask
│   │   ├── if point_size > 1.0
│   │   │   └── impls valid_mask = covered  # the coverage the image's own dilation reached, so the mask and the image it describes cannot drift apart
│   │   ├── else
│   │   │   ├── calls render_mask_from_rendering_points(rendering_points=rendering_points, valid=valid, resolution=resolution, device=rendering_points.device)
│   │   │   └── impls valid_mask = the mask it rasterized
│   │   └── return  # (seg_map, valid_mask)
│   └── else
│       └── return  # seg_map
└── def render_segmentation_from_rendering_points(rendering_points: torch.Tensor, valid: torch.Tensor, pc: PointCloud, key: str, resolution: Tuple[int, int], ignore_value: int = 255) -> torch.Tensor
    ├── # Rasterizes already-projected points into a segmentation map by writing, at each pixel, the label of the point that owns it.
    ├── assert hasattr(pc, key)  # f"PointCloud missing '{key}' field"
    ├── impls render_height, render_width = resolution
    ├── impls labels = the pc attribute named by key
    ├── assert labels.numel() > 0  # f"Labels tensor must not be empty, got {labels.numel()} elements"
    ├── impls nearest_point_index = a [render_height, render_width] tensor holding, per pixel, the index along the point axis of the valid point with the smallest depth landing there, and -1 where none landed  # reduced per pixel rather than scattered, so occlusion does not depend on which write lands last
    ├── impls pixel_labels = labels gathered at nearest_point_index
    ├── impls seg_map = pixel_labels int64-cast, with ignore_value wherever nearest_point_index is -1  # impls-node-one-step:skip
    └── return seg_map
```
