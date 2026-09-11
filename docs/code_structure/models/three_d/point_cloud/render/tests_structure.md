# `models/three_d/point_cloud/render/` tests skeleton

## Tests implementation structure

`tests/models/three_d/point_cloud/render/test_render_depth.py`

```text
test_render_depth.py
├── from typing import List, Tuple
├── import pytest
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
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
├── def test_render_depth_batched_matches_per_camera() -> None
│   ├── # A Cameras of several poses renders one cloud in a single call to [B, H, W], each slice equal to what that pose renders on its own.
│   ├── calls PointCloud(xyz=four float32 points at distinct depths)
│   ├── calls _build_cameras(focal=100.0, principal_point=50.0, translations=three distinct camera positions)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=cameras, resolution=(64, 80))
│   ├── assert the depth map is [3, 64, 80] and float32
│   ├── for each camera the batch iterates
│   │   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=that one Camera, resolution=(64, 80))
│   │   └── assert the batched map's matching slice is elementwise equal to it
│   └── return
├── def test_render_depth_batch_of_one_keeps_its_axis() -> None
│   ├── # A Cameras of length one renders to [1, H, W] rather than [H, W].
│   ├── calls PointCloud(xyz=four float32 points at distinct depths)
│   ├── calls _build_cameras(focal=100.0, principal_point=50.0, translations=one camera position)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=cameras, resolution=(64, 80))
│   ├── assert the depth map is [1, 64, 80]
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=the one Camera that batch iterates, resolution=(64, 80))
│   ├── assert that map is [64, 80] and equals the batched map's only slice
│   └── return
├── def test_render_depth_batched_cull_is_per_camera() -> None
│   ├── # Cameras seeing different subsets of one cloud each keep their own survivors, culling marking a per-camera mask rather than compacting.
│   ├── calls PointCloud(xyz=two float32 points placed so each falls inside one camera's image bounds and outside the other's)
│   ├── calls _build_cameras(focal=100.0, principal_point=50.0, translations=two camera positions offset along x)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=cameras, resolution=(64, 80), return_mask=True)
│   ├── assert both maps are [2, 64, 80] and the mask is bool
│   ├── assert each slice's mask covers pixels the other's does not
│   └── return
├── def test_render_depth_occlusion_holds_when_pixels_collide() -> None
│   ├── # A cloud dense enough that many points share a pixel still renders the nearest of them, and renders the same map every run, since occlusion must not depend on which write landed last.
│   ├── calls PointCloud(xyz=several thousand float32 points spread over a resolution small enough that most pixels take many of them)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── for each of several repeated renders
│   │   └── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(32, 32))
│   ├── assert every render returned the same map
│   └── assert each rendered pixel carries the smallest depth among the points that projected onto it
├── def test_render_depth_batched_matches_per_camera_when_pixels_collide() -> None
│   ├── # The per-camera equality the batch promises holds at that same density, which is the regime a batch is rendered at.
│   ├── calls PointCloud(xyz=several thousand float32 points spread over a resolution small enough that most pixels take many of them)
│   ├── calls _build_cameras(focal=100.0, principal_point=50.0, translations=three distinct camera positions)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=cameras, resolution=(32, 32))
│   ├── for each camera the batch iterates
│   │   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=that one Camera, resolution=(32, 32))
│   │   └── assert the batched map's matching slice is elementwise equal to it
│   └── return
├── def test_render_depth_point_size_dilates_the_rendered_discs() -> None
│   ├── # A point size above one pixel grows each rendered point into a disc, so the parameter the entry point takes changes what it returns.
│   ├── calls PointCloud(xyz=one float32 point at depth one)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=True, point_size=1.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=True, point_size=5.0)
│   └── assert the wider point size covers strictly more pixels, all at that point's own depth
├── def test_render_depth_point_size_keeps_a_nan_background() -> None
│   ├── # A NaN background survives the dilation: the discs cover the pixels they cover under a finite background, and every pixel outside them stays NaN.
│   ├── calls PointCloud(xyz=four float32 points at distinct depths)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), ignore_value=float("nan"), return_mask=True, point_size=3.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), ignore_value=-1.0, return_mask=True, point_size=3.0)
│   ├── assert the two masks are equal
│   ├── assert the NaN-background map is NaN exactly where its mask is False
│   └── assert the two maps agree wherever the mask is True
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
├── def _build_camera(focal: float, principal_point: float) -> Camera
│   ├── # Builds the identity-pose OpenGL pinhole camera on the CPU that every single-camera case here renders through.
│   ├── calls build_camera_intrinsics(model='pinhole', params=the shared focal and principal point with the extents twice that point implies, intr_convention='standard', device=torch.device('cpu'))
│   ├── calls CameraExtrinsics(extrinsics=a float32 [4, 4] identity, extr_convention='opengl', device=torch.device('cpu'))
│   ├── calls Camera(intrinsics=the intrinsics it built, extrinsics=the extrinsics it built, device=torch.device('cpu'))
│   └── return  # that camera
└── def _build_cameras(focal: float, principal_point: float, translations: List[Tuple[float, float, float]]) -> Cameras
    ├── # Builds the OpenGL pinhole batch every batched case here renders through, one pose per translation.
    ├── calls build_camera_intrinsics(model='pinhole', params=the shared focal and principal point carried once per translation, with the extents twice that point implies, intr_convention='standard', device=torch.device('cpu'))
    ├── calls CameraExtrinsics(extrinsics=a float32 [B, 4, 4] stack of identities carrying one translation each, extr_convention='opengl', device=torch.device('cpu'))
    ├── calls Cameras(intrinsics=the intrinsics it built, extrinsics=the extrinsics it built, device=torch.device('cpu'))
    └── return  # that batch
```

`tests/models/three_d/point_cloud/render/test_render_rgb.py`

```text
test_render_rgb.py
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.render import render_rgb_from_point_cloud
├── def test_render_rgb_lands_on_the_pixel_of_its_own_point() -> None
│   ├── # Each rendered pixel carries the colour of the point that projected onto it, which is the one thing a per-point attribute renderer must get right.
│   ├── calls PointCloud(xyz=three float32 points at distinct depths projecting to three separate pixels, data=one distinguishable colour per point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100))
│   └── assert each of the three pixels carries the colour belonging to the point that projected there, not another point's
├── def test_render_rgb_ignores_the_points_that_culled_out() -> None
│   ├── # A cloud with several survivors and several culled points renders only the survivors, the regime a fixture of one or two in-bounds points cannot reach.
│   ├── calls PointCloud(xyz=three float32 points inside the image bounds and two placed far outside them, data=one distinguishable colour per point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100))
│   ├── assert exactly the three survivors' pixels are painted
│   └── assert no pixel carries a culled point's colour
├── def test_render_rgb_takes_the_nearest_point_where_two_share_a_pixel() -> None
│   ├── # Two points on one ray paint the nearer one's colour, so the attribute follows the same occlusion the depth map resolves.
│   ├── calls PointCloud(xyz=two float32 points on one ray at depths three and one, data=one distinguishable colour per point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100))
│   └── assert the shared pixel carries the near point's colour
└── def _build_camera(focal: float, principal_point: float) -> Camera
    ├── # Builds the identity-pose OpenGL pinhole camera on the CPU that every case here renders through.
    ├── calls build_camera_intrinsics(model='pinhole', params=the shared focal and principal point with the extents twice that point implies, intr_convention='standard', device=torch.device('cpu'))
    ├── calls CameraExtrinsics(extrinsics=a float32 [4, 4] identity, extr_convention='opengl', device=torch.device('cpu'))
    ├── calls Camera(intrinsics=the intrinsics it built, extrinsics=the extrinsics it built, device=torch.device('cpu'))
    └── return  # that camera
```

`tests/models/three_d/point_cloud/render/test_render_segmentation.py`

```text
test_render_segmentation.py
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.render import render_segmentation_from_point_cloud
├── def test_render_segmentation_lands_on_the_pixel_of_its_own_point() -> None
│   ├── # Each rendered pixel carries the label of the point that projected onto it, which is the one thing a per-point attribute renderer must get right.
│   ├── calls PointCloud(xyz=three float32 points at distinct depths projecting to three separate pixels, data=one distinguishable label per point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_segmentation_from_point_cloud(pc=pc_data, key='labels', camera=camera, resolution=(100, 100))
│   └── assert each of the three pixels carries the label belonging to the point that projected there, not another point's
├── def test_render_segmentation_ignores_the_points_that_culled_out() -> None
│   ├── # A cloud with several survivors and several culled points renders only the survivors, the regime a fixture of one or two in-bounds points cannot reach.
│   ├── calls PointCloud(xyz=three float32 points inside the image bounds and two placed far outside them, data=one distinguishable label per point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_segmentation_from_point_cloud(pc=pc_data, key='labels', camera=camera, resolution=(100, 100))
│   ├── assert exactly the three survivors' pixels are painted
│   └── assert no pixel carries a culled point's label
├── def test_render_segmentation_takes_the_nearest_point_where_two_share_a_pixel() -> None
│   ├── # Two points on one ray paint the nearer one's label, so the attribute follows the same occlusion the depth map resolves.
│   ├── calls PointCloud(xyz=two float32 points on one ray at depths three and one, data=one distinguishable label per point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_segmentation_from_point_cloud(pc=pc_data, key='labels', camera=camera, resolution=(100, 100))
│   └── assert the shared pixel carries the near point's label
└── def _build_camera(focal: float, principal_point: float) -> Camera
    ├── # Builds the identity-pose OpenGL pinhole camera on the CPU that every case here renders through.
    ├── calls build_camera_intrinsics(model='pinhole', params=the shared focal and principal point with the extents twice that point implies, intr_convention='standard', device=torch.device('cpu'))
    ├── calls CameraExtrinsics(extrinsics=a float32 [4, 4] identity, extr_convention='opengl', device=torch.device('cpu'))
    ├── calls Camera(intrinsics=the intrinsics it built, extrinsics=the extrinsics it built, device=torch.device('cpu'))
    └── return  # that camera
```

`tests/models/three_d/point_cloud/render/test_render_normal.py`

```text
test_render_normal.py
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.render import render_normal_from_point_cloud_3d
├── def test_render_normal_lands_on_the_pixel_of_its_own_point() -> None
│   ├── # Each rendered pixel carries the normal of the point that projected onto it, which is the one thing a per-point attribute renderer must get right.
│   ├── calls PointCloud(xyz=three float32 points at distinct depths projecting to three separate pixels, data=one distinguishable normal per point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_normal_from_point_cloud_3d(pc=pc_data, camera=camera, resolution=(100, 100))
│   └── assert each of the three pixels carries the normal of the point that projected there, in the camera frame the renderer rotates it into, not another point's
├── def test_render_normal_ignores_the_points_that_culled_out() -> None
│   ├── # A cloud with several survivors and several culled points renders only the survivors, the regime a fixture of one or two in-bounds points cannot reach.
│   ├── calls PointCloud(xyz=three float32 points inside the image bounds and two placed far outside them, data=one distinguishable normal per point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_normal_from_point_cloud_3d(pc=pc_data, camera=camera, resolution=(100, 100))
│   ├── assert exactly the three survivors' pixels are painted
│   └── assert no pixel carries a culled point's normal, in the camera frame the renderer rotates it into
├── def test_render_normal_takes_the_nearest_point_where_two_share_a_pixel() -> None
│   ├── # Two points on one ray paint the nearer one's normal, so the attribute follows the same occlusion the depth map resolves.
│   ├── calls PointCloud(xyz=two float32 points on one ray at depths three and one, data=one distinguishable normal per point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_normal_from_point_cloud_3d(pc=pc_data, camera=camera, resolution=(100, 100))
│   └── assert the shared pixel carries the near point's normal, in the camera frame the renderer rotates it into
└── def _build_camera(focal: float, principal_point: float) -> Camera
    ├── # Builds the identity-pose OpenGL pinhole camera on the CPU that every case here renders through.
    ├── calls build_camera_intrinsics(model='pinhole', params=the shared focal and principal point with the extents twice that point implies, intr_convention='standard', device=torch.device('cpu'))
    ├── calls CameraExtrinsics(extrinsics=a float32 [4, 4] identity, extr_convention='opengl', device=torch.device('cpu'))
    ├── calls Camera(intrinsics=the intrinsics it built, extrinsics=the extrinsics it built, device=torch.device('cpu'))
    └── return  # that camera
```

`tests/models/three_d/point_cloud/render/test_create_circular_kernel_offsets.py`

```text
test_create_circular_kernel_offsets.py
├── import math
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.render.common.create_circular_kernel_offsets import create_circular_kernel_offsets
├── from models.three_d.point_cloud.render.render_depth import render_depth_from_point_cloud
├── def test_create_circular_kernel_offsets_disc_is_centred() -> None
│   ├── # The kernel reaches equally on both sides of the origin, since a disc that reaches farther one way grows every rendered point off its own pixel.
│   ├── for each point size of one, one and a half, two, three, four and five
│   │   ├── calls create_circular_kernel_offsets(point_size=that size, device=torch.device('cpu'))
│   │   └── assert every offset it returned has its negation in the same set
│   └── return
├── def test_create_circular_kernel_offsets_membership_is_the_radius_rule() -> None
│   ├── # The kernel is exactly the cells whose centre lies inside the disc, neither more nor fewer, checked against a radius rule the test derives itself.
│   ├── for each point size of one, one and a half, two, three, four and five
│   │   ├── calls create_circular_kernel_offsets(point_size=that size, device=torch.device('cpu'))
│   │   ├── impls expected = the integer cells of a generous search box whose distance from the origin is within half that size
│   │   └── assert the returned offsets are that set, with no cell repeated
│   └── return
├── def test_create_circular_kernel_offsets_dilates_a_point_into_a_centred_disc() -> None
│   ├── # One rendered point grows into a disc centred on its own pixel, which is the kernel's symmetry seen through the renderer that uses it.
│   ├── calls PointCloud(xyz=one float32 point at depth one)
│   ├── calls _build_camera(focal=100.0, principal_point=20.5)
│   ├── for each point size of one, one and a half, two, three, four and five
│   │   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(41, 41), return_mask=True, point_size=that size)
│   │   ├── assert the covered pixels are symmetric about the pixel the point itself landed on
│   │   └── assert the covered count is the one that size's disc holds
│   └── return
└── def _build_camera(focal: float, principal_point: float) -> Camera
    ├── # Builds the identity-pose OpenGL pinhole camera on the CPU whose own extents match the requested resolution, so no intrinsics rescaling moves the point off centre.
    ├── calls build_camera_intrinsics(model='pinhole', params=the shared focal and principal point with the extents twice that point implies, intr_convention='standard', device=torch.device('cpu'))
    ├── calls CameraExtrinsics(extrinsics=a float32 [4, 4] identity, extr_convention='opengl', device=torch.device('cpu'))
    ├── calls Camera(intrinsics=the intrinsics it built, extrinsics=the extrinsics it built, device=torch.device('cpu'))
    └── return  # that camera
```
