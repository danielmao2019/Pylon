# `models/three_d/point_cloud/render/` code skeleton

## Code implementation structure

`models/three_d/point_cloud/render/__init__.py`

```text
__init__.py
├── from models.three_d.point_cloud.render.common import apply_point_size_postprocessing, prepare_points_for_rendering, validate_rendering_inputs
├── from models.three_d.point_cloud.render.display import render_display
├── from models.three_d.point_cloud.render.render_depth import render_depth_from_point_cloud, render_depth_from_rendering_points
├── from models.three_d.point_cloud.render.render_mask import render_mask_from_rendering_points
├── from models.three_d.point_cloud.render.render_normal import render_normal_from_point_cloud_2d, render_normal_from_point_cloud_3d, render_normal_from_rendering_points_3d
├── from models.three_d.point_cloud.render.render_rgb import render_rgb_from_point_cloud, render_rgb_from_rendering_points
└── from models.three_d.point_cloud.render.render_segmentation import render_segmentation_from_point_cloud, render_segmentation_from_rendering_points
```

`models/three_d/point_cloud/render/display.py`

```text
display.py
├── from typing import Any, Dict, List, Optional, Tuple
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from models.three_d.base import BaseSceneModel
├── from models.three_d.point_cloud.render.render_rgb import render_rgb_from_point_cloud
└── def render_display(scene_model: BaseSceneModel, camera: Camera, resolution: Tuple[int, int], camera_name: Optional[str], display_cameras: Optional[List[Camera]], title: Optional[str], device: Optional[torch.device]) -> Dict[str, Any]
    ├── # Serves the viewer one titled RGB image of the scene model's point cloud at one camera, reusing a stored snapshot when the camera is named.
    ├── impls resolved_device = device if device is not None else scene_model.device
    ├── calls camera.to(resolved_device)
    ├── impls image = None
    ├── if camera_name is not None
    │   └── calls scene_model._get_snapshot(camera_name)
    ├── if image is None
    │   ├── calls render_rgb_from_point_cloud(pc=scene_model.model, camera=camera, resolution=resolution)
    │   └── if camera_name is not None
    │       └── calls scene_model._put_snapshot(camera_name, image.detach().cpu())
    ├── calls BaseSceneModel._apply_camera_overlays(image=image, display_cameras=display_cameras, render_at_camera=camera, resolution=resolution)
    ├── impls title_value = title if title is not None else the empty string
    └── return  # the {'image', 'title'} payload the display callback renders
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
│   ├── # Projects one point cloud through one camera and rasterizes its colours into an RGB image.
│   ├── calls validate_rendering_inputs(pc=pc, camera=camera, resolution=resolution, ignore_value=ignore_value, return_mask=return_mask, point_size=point_size)
│   ├── calls prepare_points_for_rendering(pc=pc, camera=camera, resolution=resolution)
│   ├── calls render_rgb_from_rendering_points(rendering_points=rendering_points, original_data_indices=original_data_indices, pc=pc, resolution=resolution, ignore_value=ignore_value)
│   ├── if point_size > 1.0
│   │   ├── calls render_depth_from_rendering_points(rendering_points=rendering_points, resolution=resolution, ignore_value=float('inf'), return_mask=False)
│   │   └── calls apply_point_size_postprocessing(rendered_image=rgb_image, depth_map=depth_map, point_size=point_size, ignore_value=float('inf'))
│   ├── if return_mask
│   │   ├── calls render_mask_from_rendering_points(rendering_points=rendering_points, resolution=resolution, device=rendering_points.device)
│   │   ├── if point_size > 1.0
│   │   │   ├── calls render_depth_from_rendering_points(rendering_points=rendering_points, resolution=resolution, ignore_value=float('inf'), return_mask=False)
│   │   │   └── calls apply_point_size_postprocessing(rendered_image=valid_mask.float(), depth_map=depth_map, point_size=point_size, ignore_value=float('inf'))
│   │   └── return  # (rgb_image, valid_mask)
│   └── else
│       └── return  # rgb_image
└── def render_rgb_from_rendering_points(rendering_points: torch.Tensor, original_data_indices: torch.Tensor, pc: PointCloud, resolution: Tuple[int, int], ignore_value: float = 0.0) -> torch.Tensor
    ├── # Rasterizes the point cloud's per-point colours into a [3, H, W] image at the pixels the already-projected points landed on.
    ├── impls colors = a clone of pc.rgb
    ├── if colors is an integer dtype or its range sits in [0, 255]
    │   └── impls divide colors by 255.0
    ├── impls clamp colors into [0, 1]
    ├── impls pixel_colors = colors gathered at original_data_indices
    ├── impls rgb_image = a full [3, H, W] float32 tensor of ignore_value on the rendering_points device
    ├── impls scatter pixel_colors into rgb_image at the (y, x) integer pixel coordinates in rendering_points columns 1/0
    └── return  # rgb_image [3, H, W] with values in [0, 1]
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
│   ├── # Projects one point cloud through one camera and rasterizes the surviving points' depths into a depth map.
│   ├── calls validate_rendering_inputs(pc=pc, camera=camera, resolution=resolution, ignore_value=ignore_value, return_mask=return_mask, point_size=point_size)
│   ├── calls prepare_points_for_rendering(pc=pc, camera=camera, resolution=resolution)
│   ├── calls render_depth_from_rendering_points(rendering_points=rendered_points, resolution=resolution, ignore_value=ignore_value, return_mask=return_mask)
│   └── return  # the depth map, paired with the valid mask when return_mask
└── def render_depth_from_rendering_points(rendering_points: torch.Tensor, resolution: Tuple[int, int], ignore_value: float = float('inf'), return_mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ├── # Rasterizes the camera-space depth column of already-projected points into an [H, W] map whose empty pixels carry ignore_value.
    ├── impls depth_map = a full [H, W] float32 tensor of ignore_value on the rendering_points device
    ├── impls scatter the depth column into depth_map at the (y, x) integer pixel coordinates in rendering_points columns 1/0
    ├── if return_mask
    │   ├── calls render_mask_from_rendering_points(rendering_points=rendering_points, resolution=resolution, device=rendering_points.device)
    │   └── return  # (depth_map, valid_mask)
    └── else
        └── return  # depth_map
```

`models/three_d/point_cloud/render/render_mask.py`

```text
render_mask.py
├── from typing import Tuple
├── import torch
└── def render_mask_from_rendering_points(rendering_points: torch.Tensor, resolution: Tuple[int, int], device: torch.device) -> torch.Tensor
    ├── # Marks which pixels of the image any projected point landed on, the coverage every render entry point pairs its output with.
    ├── impls valid_mask = a zeros [H, W] bool tensor on device
    ├── impls set valid_mask True at the (y, x) integer pixel coordinates in rendering_points columns 1/0
    └── return  # valid_mask [H, W]
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
├── from models.three_d.point_cloud.render.render_depth import render_depth_from_point_cloud, render_depth_from_rendering_points
├── from models.three_d.point_cloud.render.render_mask import render_mask_from_rendering_points
├── from utils.conversions.depth_to_normals import depth_to_normals
├── def render_normal_from_point_cloud_2d(pc: PointCloud, camera: Camera, resolution: Tuple[int, int], ignore_value: float = 0.0, return_mask: bool = False, point_size: float = 1.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
│   ├── # Derives the normal map from the rendered depth's gradients, so a point cloud carrying no normals still yields one.
│   ├── calls render_depth_from_point_cloud(pc=pc, camera=camera, resolution=resolution, ignore_value=float('inf'), return_mask=False, point_size=point_size)
│   ├── impls intrinsics_matrix = the [3, 3] pinhole matrix built from camera.intrinsics fx/fy/cx/cy
│   ├── calls depth_to_normals(depth_map=depth_map, camera_intrinsics=intrinsics_matrix, depth_ignore_value=float('inf'), normal_ignore_value=ignore_value, return_mask=return_mask)
│   └── return  # the opencv-frame normal map, paired with the valid mask when return_mask
├── def render_normal_from_point_cloud_3d(pc: PointCloud, camera: Camera, resolution: Tuple[int, int], ignore_value: float = 0.0, return_mask: bool = False, point_size: float = 1.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
│   ├── # Projects one point cloud through one camera and rasterizes its stored normals, the exact-normal alternative to the depth-gradient path.
│   ├── calls validate_rendering_inputs(pc=pc, camera=camera, resolution=resolution, ignore_value=ignore_value, return_mask=return_mask, point_size=point_size)
│   ├── calls prepare_points_for_rendering(pc=pc, camera=camera, resolution=resolution)
│   ├── calls render_normal_from_rendering_points_3d(rendering_points=rendering_points, original_data_indices=original_data_indices, pc_data=pc, camera=camera, resolution=resolution, ignore_value=ignore_value)
│   ├── if point_size > 1.0
│   │   ├── calls render_depth_from_rendering_points(rendering_points=rendering_points, resolution=resolution, ignore_value=float('inf'), return_mask=False)
│   │   ├── calls apply_point_size_postprocessing(rendered_image=normal_map, depth_map=depth_map, point_size=point_size, ignore_value=float('inf'))
│   │   └── impls re-normalize the dilated normal_map over the pixels that are not ignore_value
│   ├── if return_mask
│   │   ├── calls render_mask_from_rendering_points(rendering_points=rendering_points, resolution=resolution, device=rendering_points.device)
│   │   ├── if point_size > 1.0
│   │   │   ├── calls render_depth_from_rendering_points(rendering_points=rendering_points, resolution=resolution, ignore_value=float('inf'), return_mask=False)
│   │   │   └── calls apply_point_size_postprocessing(rendered_image=valid_mask.float(), depth_map=depth_map, point_size=point_size, ignore_value=float('inf'))
│   │   └── return  # (normal_map, valid_mask)
│   └── else
│       └── return  # normal_map
└── def render_normal_from_rendering_points_3d(rendering_points: torch.Tensor, original_data_indices: torch.Tensor, pc_data: PointCloud, camera: Camera, resolution: Tuple[int, int], ignore_value: float = 0.0) -> torch.Tensor
    ├── # Rasterizes the point cloud's own world-space normals, rotated into the opencv camera frame, at the pixels the already-projected points landed on.
    ├── impls world_normals = pc_data.normals unit-normalized along the last dim
    ├── impls visible_world_normals = world_normals gathered at original_data_indices
    ├── calls camera.to(device=rendering_points.device, extr_convention="opencv")
    ├── impls camera_normals = visible_world_normals rotated by the w2c rotation block, then re-normalized
    ├── impls normal_map = a full [3, H, W] float32 tensor of ignore_value on the rendering_points device
    ├── impls scatter camera_normals into normal_map at the (y, x) integer pixel coordinates in rendering_points columns 1/0
    └── return  # normal_map [3, H, W]
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
│   ├── # Projects one point cloud through one camera and rasterizes the labels under key into a segmentation map.
│   ├── calls validate_rendering_inputs(pc=pc, camera=camera, resolution=resolution, ignore_value=ignore_value, return_mask=return_mask, point_size=point_size)
│   ├── calls prepare_points_for_rendering(pc=pc, camera=camera, resolution=resolution)
│   ├── calls render_segmentation_from_rendering_points(rendering_points=rendering_points, original_data_indices=original_data_indices, pc=pc, key=key, resolution=resolution, ignore_value=ignore_value)
│   ├── if point_size > 1.0
│   │   ├── calls render_depth_from_rendering_points(rendering_points=rendering_points, resolution=resolution, ignore_value=float('inf'), return_mask=False)
│   │   └── calls apply_point_size_postprocessing(rendered_image=seg_map.float(), depth_map=depth_map, point_size=point_size, ignore_value=float('inf'))
│   ├── if return_mask
│   │   ├── calls render_mask_from_rendering_points(rendering_points=rendering_points, resolution=resolution, device=rendering_points.device)
│   │   ├── if point_size > 1.0
│   │   │   ├── calls render_depth_from_rendering_points(rendering_points=rendering_points, resolution=resolution, ignore_value=float('inf'), return_mask=False)
│   │   │   └── calls apply_point_size_postprocessing(rendered_image=valid_mask.float(), depth_map=depth_map, point_size=point_size, ignore_value=float('inf'))
│   │   └── return  # (seg_map, valid_mask)
│   └── else
│       └── return  # seg_map
└── def render_segmentation_from_rendering_points(rendering_points: torch.Tensor, original_data_indices: torch.Tensor, pc: PointCloud, key: str, resolution: Tuple[int, int], ignore_value: int = 255) -> torch.Tensor
    ├── # Rasterizes the per-point labels held under one point cloud field into an [H, W] integer label map.
    ├── impls labels = the pc field named by key
    ├── impls pixel_labels = labels gathered at original_data_indices
    ├── impls seg_map = a full [H, W] int64 tensor of ignore_value on the rendering_points device
    ├── impls scatter pixel_labels into seg_map at the (y, x) integer pixel coordinates in rendering_points columns 1/0
    └── return  # seg_map [H, W] of int64 labels
```

`models/three_d/point_cloud/render/render_rgb_o3d.py`

```text
render_rgb_o3d.py
├── from typing import Tuple
├── import numpy as np
├── import open3d as o3d
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
└── def render_rgb_from_pointcloud_o3d(pc_data: PointCloud, camera: Camera, resolution: Tuple[int, int]) -> torch.Tensor
    ├── # Renders a coloured point cloud through Open3D's offscreen rasterizer, the third-party alternative to the in-repo projection path.
    ├── calls camera.to(device=points.device, extr_convention="opengl")
    ├── impls camera_intrinsics = the [3, 3] pinhole matrix built from camera.intrinsics fx/fy/cx/cy
    ├── impls infer the native (width, height) as twice the principal point
    ├── impls scale camera_intrinsics fx/fy/cx/cy by the render-over-native ratio
    ├── if pc_data.rgb is an integer dtype and its range sits in [0, 255]
    │   └── impls divide colors by 255.0
    ├── impls pcd = an o3d PointCloud carrying the cpu numpy points and colors
    ├── impls renderer = an o3d OffscreenRenderer at the render resolution, on a black background
    ├── impls add pcd to the renderer scene under a defaultUnlit material of point size 2.0
    ├── impls set the renderer camera projection from camera_intrinsics with near 0.1 and far 1000.0
    ├── impls aim the renderer camera by look_at from the extrinsics center, forward and up
    ├── impls img_array = the rendered image as a numpy [H, W, 3] uint8 array
    └── return  # img_array scaled to [0, 1] and permuted to a [3, H, W] float32 tensor
```

`models/three_d/point_cloud/render/render_rgb_volumetric.py`

```text
render_rgb_volumetric.py
├── import itertools
├── import json
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
├── from data.structures.three_d.nerfstudio.nerfstudio_data import NerfStudio_Data
├── from data.structures.three_d.point_cloud.io.save_point_cloud import save_point_cloud
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.structures.three_d.point_cloud.select import Select
├── from models.three_d.point_cloud.render.common.prepare_points_for_rendering import prepare_points_for_rendering
├── from models.three_d.point_cloud.render.render_rgb import render_rgb_from_point_cloud
├── from models.three_d.splatfacto.load_splatfacto import load_splatfacto_model
├── from models.three_d.splatfacto.render import render_rgb_from_splatfacto
├── def render_rgb_from_point_cloud_volumetric(pc: PointCloud, camera: Camera, resolution: Tuple[int, int], debug: bool = False) -> torch.Tensor
│   ├── # Renders a point cloud volumetrically by training a splatfacto model on views rendered from it, then evaluating that model at the requested camera.
│   ├── impls downscale_factor = the valid factor of {1, 2, 4, 8} matching the render-over-native resolution ratio
│   ├── calls prepare_points_for_rendering(pc=pc, camera=camera, resolution=resolution)
│   ├── calls Select(indices=image_plane_points_indices)
│   ├── calls gen_auxiliary_cameras(points=pc.xyz, camera=camera)
│   ├── impls train_extrinsics = the primary extrinsics followed by every auxiliary camera's
│   ├── for _extrinsics in train_extrinsics
│   │   ├── calls Camera(intrinsics=intrinsics, extrinsics=_extrinsics, device=pc.device)
│   │   ├── calls render_rgb_from_point_cloud(pc=pc, camera=render_camera, resolution=resolution, return_mask=True)
│   │   └── impls append the image and mask to the training set
│   ├── if debug
│   │   └── impls tempdir = the retained ./test_volumetric_rendering workspace, with cleanup_fn = None
│   ├── else
│   │   └── impls temp_dir_context = a TemporaryDirectory whose path is tempdir and whose cleanup is cleanup_fn
│   ├── try
│   │   ├── calls _create_images(images=images, output_root=tempdir, downscale_factor=downscale_factor)
│   │   ├── calls _create_masks(masks=masks, output_root=tempdir, downscale_factor=downscale_factor)
│   │   ├── calls _create_ply(pc=pc, output_root=tempdir)
│   │   ├── calls _create_nerfstudio(intrinsics=intrinsics, train_extrinsics=train_extrinsics, eval_extrinsics=extrinsics, convention=convention, output_root=tempdir)
│   │   ├── calls _run_ns_train_splatfacto(dataset_root=dataset_root, downscale_factor=downscale_factor)
│   │   ├── calls _assert_checkpoint_exists(model_dir=model_dir)
│   │   ├── calls load_splatfacto_model(model_dir=str(model_dir), device=target_device)
│   │   └── calls render_rgb_from_splatfacto(model=pipeline, camera=camera, resolution=resolution)
│   ├── finally
│   │   └── if cleanup_fn is not None
│   │       └── calls temp_dir_context.cleanup()
│   └── return  # the evaluation render moved onto the point cloud's device
├── def gen_auxiliary_cameras(points: torch.Tensor, camera: Camera) -> List[Camera]
│   ├── # Surrounds the primary camera with 26 nearby viewpoints, so the splat training set covers more than the one view being rendered.
│   ├── calls camera.to(device=device, extr_convention="standard")
│   ├── impls step = half the distance from the camera position to the point cloud centroid
│   ├── impls direction_specs = the 26 unit vectors over itertools.product([-1, 0, 1], repeat=3) minus the origin
│   ├── for direction_unit in direction_specs
│   │   ├── impls aux_standard = the primary rotation block with the position translated by direction_unit * step
│   │   ├── calls Camera(intrinsics=camera.intrinsics, extrinsics=CameraExtrinsics(extrinsics=aux_standard, extr_convention="standard", device=device), device=device)
│   │   └── impls append the auxiliary camera converted back to the primary camera's convention
│   └── return  # the 26 auxiliary cameras, in the primary camera's own convention
├── def _create_images(images: List[torch.Tensor], output_root: str, downscale_factor: int) -> None
│   ├── # Writes the rendered RGB tensors into the nerfstudio dataset's downscale-suffixed images directory.
│   └── for idx, image in enumerate(images)
│       └── impls save the clamped [3, H, W] tensor as an 8-bit PNG named image_{idx:02d}.png
├── def _create_masks(masks: List[torch.Tensor], output_root: str, downscale_factor: int) -> None
│   ├── # Writes the rendered coverage masks into the nerfstudio dataset's downscale-suffixed masks directory.
│   └── for idx, mask in enumerate(masks)
│       └── impls save the boolean [H, W] mask as an 8-bit grayscale PNG named mask_{idx:02d}.png
├── def _create_ply(pc: PointCloud, output_root: str) -> None
│   ├── # Writes the visibility-filtered point cloud as the dataset's point_cloud.ply seed geometry.
│   └── calls save_point_cloud(pc, str(ply_path))
├── def _create_nerfstudio(cameras: List[Camera], output_root: Path) -> None
│   ├── # Writes the transforms.json that binds the rendered frames, the seed ply, and the training cameras into one nerfstudio dataset.
│   ├── impls intrinsic_params = the fl_x / fl_y / cx / cy plus zeroed distortion terms read off the first camera
│   ├── impls resolution = twice the first camera's principal point, as (height, width)
│   ├── impls applied_transform = the fixed [3, 4] axis permutation nerfstudio expects
│   ├── calls Cameras(intrinsics=[camera.intrinsics for camera in cameras], extrinsics=[camera.extrinsics for camera in cameras], names=camera_names, ids=[camera.id for camera in cameras], device=cameras[0].device)
│   ├── calls NerfStudio_Data(data=payload, device=cameras[0].device, intrinsic_params=intrinsic_params, resolution=resolution, camera_model=camera_model, intrinsics=intrinsics, applied_transform=applied_transform, ply_file_path="point_cloud.ply", cameras=nerfstudio_cameras, modalities=modalities, train_filenames=None, val_filenames=None, test_filenames=None)
│   └── calls nerfstudio_data.save(output_path=nerfstudio_path)
├── def _run_ns_train_splatfacto(dataset_root: Path, downscale_factor: int) -> Path
│   ├── # Trains splatfacto on the written dataset by shelling out to ns-train, and locates the run directory it produced.
│   ├── impls ns_train_cmd = the ns-train splatfacto argv over dataset_root at a 1.0 train split
│   ├── calls subprocess.run(ns_train_cmd, check=True)
│   └── return  # the parent of the newest config.yml under the output directory
└── def _assert_checkpoint_exists(model_dir: Path) -> Path
    ├── # Asserts the 30K-iteration checkpoint the evaluation render depends on was actually reached.
    ├── impls assert the step-000029999.ckpt file exists under model_dir / "nerfstudio_models"
    └── return  # the checkpoint path
```

`models/three_d/point_cloud/render/common/__init__.py`

```text
__init__.py
├── from models.three_d.point_cloud.render.common.apply_point_size_postprocessing import apply_point_size_postprocessing
├── from models.three_d.point_cloud.render.common.create_circular_kernel_offsets import create_circular_kernel_offsets
├── from models.three_d.point_cloud.render.common.prepare_points_for_rendering import prepare_points_for_rendering
└── from models.three_d.point_cloud.render.common.validate_rendering_inputs import validate_rendering_inputs
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

`models/three_d/point_cloud/render/common/validate_rendering_inputs.py`

```text
validate_rendering_inputs.py
├── from typing import Optional, Tuple, Union
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
└── def validate_rendering_inputs(pc: PointCloud, camera: Camera, resolution: Tuple[int, int], ignore_value: Optional[Union[int, float]] = None, return_mask: bool = False, point_size: float = 1.0) -> None
    ├── # Asserts the preconditions every point cloud rendering entry point shares, so each one states them once by calling here.
    ├── impls assert pc is a PointCloud and camera is a Camera
    ├── impls assert the camera intrinsics and extrinsics sit on the same device as pc.xyz
    ├── impls assert resolution is a 2-element tuple or list of positive ints
    ├── if ignore_value is not None
    │   └── impls assert ignore_value is an int or a float
    ├── impls assert return_mask is a bool
    └── impls assert point_size is numeric and at least 1.0
```

`models/three_d/point_cloud/render/common/apply_point_size_postprocessing.py`

```text
apply_point_size_postprocessing.py
├── from typing import Union
├── import torch
├── from models.three_d.point_cloud.render.common.create_circular_kernel_offsets import create_circular_kernel_offsets
└── def apply_point_size_postprocessing(rendered_image: torch.Tensor, depth_map: torch.Tensor, point_size: float, ignore_value: Union[int, float] = 0.0) -> torch.Tensor
    ├── # Grows every rendered point to the requested diameter by depth-aware dilation, so the nearest surface wins each contested pixel.
    ├── if point_size <= 1.0
    │   └── return  # rendered_image unchanged
    ├── impls result = a clone of rendered_image
    ├── calls create_circular_kernel_offsets(point_size, device)
    ├── impls y_coords, x_coords = the [H, W] pixel coordinate grids
    ├── for dy, dx in kernel_offsets
    │   ├── impls neighbor_y, neighbor_x = the coordinate grids shifted by (dy, dx)
    │   ├── impls valid_mask = the in-image-bounds test over the shifted coordinates
    │   ├── if not valid_mask.any()
    │   │   └── continue
    │   ├── impls restrict the current and neighbour coordinates to valid_mask
    │   ├── impls propagate_mask = the neighbour carries depth and is nearer than the current pixel
    │   └── if propagate_mask.any()
    │       ├── impls restrict both coordinate sets to propagate_mask
    │       ├── if is_multichannel
    │       │   └── impls copy rendered_image's [:, neighbour] pixels into result's [:, current] pixels
    │       └── else
    │           └── impls copy rendered_image's [neighbour] pixels into result's [current] pixels
    └── return  # result, the dilated image carrying rendered_image's shape
```

`models/three_d/point_cloud/render/common/create_circular_kernel_offsets.py`

```text
create_circular_kernel_offsets.py
├── import torch
└── def create_circular_kernel_offsets(point_size: float, device: torch.device) -> torch.Tensor
    ├── # Enumerates the (y, x) offsets of the centred disc of the given diameter, the neighbourhood a rendered point is dilated over.
    ├── impls kernel_size = the ceiling of point_size, kernel_radius = half of point_size
    ├── impls y_kernel, x_kernel = the meshgrid over kernel_size centred on zero
    ├── impls circular_mask = the offsets whose euclidean distance is within kernel_radius
    └── return  # the [num_pixels_in_circle, 2] tensor of (y, x) offsets inside the disc
```
