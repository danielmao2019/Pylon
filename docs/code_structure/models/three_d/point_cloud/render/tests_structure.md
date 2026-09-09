# `models/three_d/point_cloud/render/` tests skeleton

## Tests implementation structure

`tests/models/three_d/point_cloud/render/test_render_depth.py`

```text
test_render_depth.py
├── import pytest
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.render import render_depth_from_point_cloud
├── def test_render_depth_basic() -> None
│   ├── # A depth map asked for without a mask comes back at the requested resolution, in float32, with every rendered pixel at a positive depth.
│   ├── calls PointCloud(xyz=four float32 points at depths one, two, one and a half, and three)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=False)
│   ├── assert the depth map is [100, 100] and float32
│   └── assert every depth other than the -1.0 background is positive
├── def test_render_depth_with_mask() -> None
│   ├── # The mask a caller asks for marks exactly the rendered pixels, and the background carries the ignore value on the rest.
│   ├── calls PointCloud(xyz=three float32 points at increasing depth)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=True)
│   ├── assert both maps are [100, 100] and the mask is bool
│   ├── assert the mask covers some pixels but not the whole image
│   └── assert depths are positive under the mask and -1.0 outside it
├── def test_render_depth_sorting() -> None
│   ├── # Two points on one ray render as the near one, since a farther point must not overwrite what occludes it.
│   ├── calls PointCloud(xyz=two float32 points on one ray, at depths three and one)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100))
│   ├── impls valid_depths = the depths other than the -1.0 background
│   └── if valid_depths is non-empty
│       └── assert the smallest of them is below one and a half
├── def test_render_depth_custom_ignore_value() -> None
│   ├── # The background value is the caller's to name, and it reaches every pixel no point projected onto.
│   ├── calls PointCloud(xyz=one float32 point at depth one)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), ignore_value=-999.0)
│   └── assert over nine tenths of the pixels carry that value
├── def test_render_depth_points_behind_camera() -> None
│   ├── # A point behind an OpenGL camera is dropped rather than folded back in front of it.
│   ├── calls PointCloud(xyz=one float32 point behind the camera and one in front)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=True)
│   └── assert the mask covers at least one pixel and every depth under it is positive
├── def test_render_depth_multiple_points_per_pixel() -> None
│   ├── # Several points landing on one pixel resolve to the nearest, which is the same rule sorting states at pixel granularity.
│   ├── calls PointCloud(xyz=four float32 points within a hundredth of one ray, at four depths)
│   ├── calls _build_camera(focal=1000.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100))
│   ├── impls valid_depths = the depths of the central four-by-four region other than the -1.0 background
│   └── if valid_depths is non-empty
│       └── assert the smallest of them is below one and a half
├── def test_render_depth_intrinsics_scaling() -> None
│   ├── # One camera renders at two resolutions, so the intrinsics are scaled to the request rather than pinned to the camera's own extents.
│   ├── calls PointCloud(xyz=two float32 points at depths one and two)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(50, 50))
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(200, 200))
│   ├── assert each map has the resolution it was asked for
│   └── assert each carries at least one rendered pixel
├── def test_render_depth_invalid_inputs() -> None
│   ├── # The malformed inputs are refused where each is first named, which for the two camera pieces is their own construction rather than the render call.
│   ├── calls PointCloud(xyz=one float32 point at depth one)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls render_depth_from_point_cloud(pc='not a point cloud', camera=valid_camera, resolution=(100, 100))
│   ├── with pytest.raises(AssertionError)
│   │   └── calls Camera(intrinsics=a float32 [4, 4] identity, extrinsics=an opengl CameraExtrinsics, device=torch.device('cpu'))
│   ├── with pytest.raises(AssertionError)
│   │   └── calls CameraExtrinsics(extrinsics=a float32 [3, 3] identity, extr_convention='opengl', device=torch.device('cpu'))
│   └── with pytest.raises(AssertionError)
│       └── calls render_depth_from_point_cloud(pc=valid_pc_data, camera=valid_camera, resolution=(0, 100))
├── def test_a_float64_cloud_renders_against_a_float32_camera() -> None
│   ├── # No load forces f4 any more, so the camera is brought to the coordinates rather than the projection raising on mismatched widths.
│   ├── calls PointCloud(xyz=four float64 points at increasing depth)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100))
│   └── assert the depth map is [100, 100] and every depth other than the -1.0 background is finite
├── def test_the_coordinates_are_not_narrowed_to_the_camera() -> None
│   ├── # Narrowing the points to meet the camera would throw away the precision the double-precision path exists to keep.
│   ├── calls PointCloud(xyz=two float64 points whose depths differ only past float32's precision)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100))
│   └── assert the two points land at distinct depths
└── def _build_camera(focal: float, principal_point: float) -> Camera
    ├── # Builds the identity-pose OpenGL pinhole camera on the CPU that every case here renders through.
    ├── calls build_camera_intrinsics(model='pinhole', params=the shared focal and principal point with the extents twice that point implies, intr_convention='standard', device=torch.device('cpu'))
    ├── calls CameraExtrinsics(extrinsics=a float32 [4, 4] identity, extr_convention='opengl', device=torch.device('cpu'))
    ├── calls Camera(intrinsics=the intrinsics it built, extrinsics=the extrinsics it built, device=torch.device('cpu'))
    └── return  # that camera
```
