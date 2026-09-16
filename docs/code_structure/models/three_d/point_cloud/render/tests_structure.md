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
│   └── for each camera the batch iterates
│       ├── calls render_depth_from_point_cloud(pc=pc_data, camera=that one Camera, resolution=(64, 80))
│       └── assert the batched map's matching slice is elementwise equal to it
├── def test_render_depth_batch_of_one_keeps_its_axis() -> None
│   ├── # A Cameras of length one renders to [1, H, W] rather than [H, W].
│   ├── calls PointCloud(xyz=four float32 points at distinct depths)
│   ├── calls _build_cameras(focal=100.0, principal_point=50.0, translations=one camera position)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=cameras, resolution=(64, 80))
│   ├── assert the depth map is [1, 64, 80]
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=the one Camera that batch iterates, resolution=(64, 80))
│   └── assert that map is [64, 80] and equals the batched map's only slice
├── def test_render_depth_batched_cull_is_per_camera() -> None
│   ├── # Cameras seeing different subsets of one cloud each keep their own survivors, culling marking a per-camera mask rather than compacting.
│   ├── calls PointCloud(xyz=two float32 points placed so each falls inside one camera's image bounds and outside the other's)
│   ├── calls _build_cameras(focal=100.0, principal_point=50.0, translations=two camera positions offset along x)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=cameras, resolution=(64, 80), return_mask=True)
│   ├── assert both maps are [2, 64, 80] and the mask is bool
│   └── assert each slice's mask covers pixels the other's does not
├── def test_render_depth_occlusion_holds_when_pixels_collide() -> None
│   ├── # A cloud dense enough that many points share a pixel still renders the nearest of them, and renders the same map every run, since occlusion must not depend on which write landed last.
│   ├── impls focal = 100.0
│   ├── impls principal_point = 50.0
│   ├── impls resolution = (32, 32)
│   ├── impls render_height, render_width = resolution
│   ├── impls render_fx = focal * render_width / (2.0 * principal_point)  # the camera's own extents are twice its principal point, so rendering at this resolution restates its intrinsics by that ratio
│   ├── impls render_fy = focal * render_height / (2.0 * principal_point)
│   ├── impls render_cx = principal_point * render_width / (2.0 * principal_point)
│   ├── impls render_cy = principal_point * render_height / (2.0 * principal_point)
│   ├── impls num_points = 4096  # several thousand points into 1024 pixels, so most pixels take several of them
│   ├── impls generator = a torch.Generator seeded with 0
│   ├── impls target_columns = num_points integers drawn uniformly from [0, render_width) with generator
│   ├── impls target_rows = num_points integers drawn uniformly from [0, render_height) with generator
│   ├── impls depths = 1.0 + 3.0 * num_points uniform [0, 1) draws with generator
│   ├── calls PointCloud(xyz=the [num_points, 3] stack of (target_columns + 0.5 - render_cx) * depths / render_fx, -(target_rows + 0.5 - render_cy) * depths / render_fy and -depths)  # -> pc_data, each point aimed at the centre of its drawn pixel
│   ├── calls _build_camera(focal=focal, principal_point=principal_point)  # -> camera
│   ├── impls depth_maps = an empty list
│   ├── for each of six repetitions
│   │   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=resolution)  # -> depth_map
│   │   └── impls depth_maps gains depth_map
│   ├── for render_index, depth_map in enumerate(depth_maps)
│   │   └── assert torch.equal(depth_map, depth_maps[0])  # "Repeated renders of one camera must give the same depth map.", reporting render_index and the max abs difference from the first render
│   ├── impls point_depths = -pc_data.xyz[:, 2]  # the nearest depth per pixel is projected here rather than read back from the render
│   ├── impls point_columns = pc_data.xyz[:, 0] / point_depths * render_fx + render_cx
│   ├── impls point_rows = -pc_data.xyz[:, 1] / point_depths * render_fy + render_cy
│   ├── impls inside = point_depths > 0, with point_columns in [0, render_width) and point_rows in [0, render_height)
│   ├── impls expected_depth_map = a float32 map of shape resolution filled with -1.0
│   ├── impls points_per_pixel = an int64 map of shape resolution filled with 0
│   ├── for point_row, point_column, point_depth in zip(point_rows[inside] cast to int64 as a list, point_columns[inside] cast to int64 as a list, point_depths[inside] as a list, strict=True)
│   │   ├── if points_per_pixel[point_row, point_column] == 0 or point_depth < expected_depth_map[point_row, point_column]
│   │   │   └── impls expected_depth_map[point_row, point_column] = point_depth
│   │   └── impls points_per_pixel[point_row, point_column] += 1
│   └── assert torch.equal(depth_maps[0], expected_depth_map)  # "Each rendered pixel must carry the smallest depth among the points that projected onto it.", reporting the max abs difference and the differing-pixel count
├── def test_render_depth_batched_matches_per_camera_when_pixels_collide() -> None
│   ├── # The per-camera equality the batch promises holds at that same density, which is the regime a batch is rendered at.
│   ├── impls focal = 100.0
│   ├── impls principal_point = 50.0
│   ├── impls resolution = (32, 32)
│   ├── impls render_height, render_width = resolution
│   ├── impls render_fx = focal * render_width / (2.0 * principal_point)
│   ├── impls render_fy = focal * render_height / (2.0 * principal_point)
│   ├── impls render_cx = principal_point * render_width / (2.0 * principal_point)
│   ├── impls render_cy = principal_point * render_height / (2.0 * principal_point)
│   ├── impls num_points = 4096
│   ├── impls generator = a torch.Generator seeded with 0
│   ├── impls target_columns = num_points integers drawn uniformly from [0, render_width) with generator
│   ├── impls target_rows = num_points integers drawn uniformly from [0, render_height) with generator
│   ├── impls depths = 1.0 + 3.0 * num_points uniform [0, 1) draws with generator
│   ├── calls PointCloud(xyz=the [num_points, 3] stack of (target_columns + 0.5 - render_cx) * depths / render_fx, -(target_rows + 0.5 - render_cy) * depths / render_fy and -depths)  # -> pc_data; several thousand float32 points over a resolution small enough that most pixels take many of them
│   ├── calls _build_cameras(focal=focal, principal_point=principal_point, translations=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.15, 0.0)])  # -> cameras, three distinct camera positions
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=cameras, resolution=resolution)  # -> depth_maps
│   └── for index, camera in enumerate(cameras)
│       ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=resolution)  # -> depth_map
│       └── assert torch.equal(depth_maps[index], depth_map)  # "Each batched slice must equal what its own camera renders alone.", reporting index, the max abs difference and the differing-pixel count
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
    ├── impls batch_size = len(translations)
    ├── impls extrinsics = a float32 [batch_size, 4, 4] stack of identities
    ├── impls extrinsics[:, :3, 3] = translations as a float32 tensor  # one translation per identity pose
    ├── calls build_camera_intrinsics(model='pinhole', params=fx and fy of focal, cx and cy of principal_point, and h and w of 2.0 * principal_point, each a [batch_size] tensor, intr_convention='standard', device=torch.device('cpu'))
    ├── calls CameraExtrinsics(extrinsics=extrinsics, extr_convention='opengl', device=torch.device('cpu'))
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
│   ├── calls PointCloud(xyz=three float32 points at depths one, two and four projecting to three separate pixels, data=one distinguishable colour per point)  # -> pc_data
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)  # -> camera
│   ├── impls resolution = (100, 100)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=resolution)  # -> rgb_image
│   ├── impls pixels = [(50, 50), (43, 62), (56, 37)]  # the pixels the three points project to, in point order
│   └── for point_index, (row, column) in enumerate(pixels)
│       └── assert torch.equal(rgb_image[:, row, column], pc_data.rgb[point_index])  # "Each rendered pixel must carry the colour of the point that projected onto it.", reporting point_index, row, column and both colours; the colour belonging to the point that projected there, not another point's
├── def test_render_rgb_ignores_the_points_that_culled_out() -> None
│   ├── # A cloud with several survivors and several culled points renders only the survivors, the regime a fixture of one or two in-bounds points cannot reach.
│   ├── calls PointCloud(xyz=three float32 points inside the image bounds and two placed far outside them, data=one distinguishable colour per point)  # -> pc_data
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)  # -> camera
│   ├── impls resolution = (100, 100)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=resolution)  # -> rgb_image
│   ├── impls painted = the pixels where any channel of rgb_image differs from 0.0  # 0.0 is the renderer's default ignore_value
│   ├── impls expected_painted = a bool map of shape resolution filled with False
│   ├── for row, column in [(50, 50), (43, 62), (56, 37)]
│   │   └── impls expected_painted[row, column] = True
│   ├── assert torch.equal(painted, expected_painted)  # "Exactly the three survivors' pixels must be painted.", reporting both painted counts and the painted pixels
│   └── for culled_index in (3, 4)
│       ├── impls culled_color = pc_data.rgb[culled_index] reshaped to [3, 1, 1]
│       └── assert not (rgb_image == culled_color).all(dim=0).any()  # "No pixel may carry the colour of a point that culled out.", reporting culled_index, its colour and the pixels carrying it
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
│   ├── calls PointCloud(xyz=three float32 points at depths one, two and four projecting to three separate pixels, data=one distinguishable label per point)  # -> pc_data
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)  # -> camera
│   ├── impls resolution = (100, 100)
│   ├── calls render_segmentation_from_point_cloud(pc=pc_data, key='labels', camera=camera, resolution=resolution)  # -> seg_map
│   ├── impls pixels = [(50, 50), (43, 62), (56, 37)]  # the pixels the three points project to, in point order
│   └── for point_index, (row, column) in enumerate(pixels)
│       └── assert seg_map[row, column] == pc_data.labels[point_index]  # "Each rendered pixel must carry the label of the point that projected onto it.", reporting point_index, row, column and both labels; the label belonging to the point that projected there, not another point's
├── def test_render_segmentation_ignores_the_points_that_culled_out() -> None
│   ├── # A cloud with several survivors and several culled points renders only the survivors, the regime a fixture of one or two in-bounds points cannot reach.
│   ├── calls PointCloud(xyz=three float32 points inside the image bounds and two placed far outside them, data=one distinguishable label per point)  # -> pc_data
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)  # -> camera
│   ├── impls resolution = (100, 100)
│   ├── calls render_segmentation_from_point_cloud(pc=pc_data, key='labels', camera=camera, resolution=resolution)  # -> seg_map
│   ├── impls painted = seg_map != 255  # 255 is the renderer's default ignore_value
│   ├── impls expected_painted = a bool map of shape resolution filled with False
│   ├── for row, column in [(50, 50), (43, 62), (56, 37)]
│   │   └── impls expected_painted[row, column] = True
│   ├── assert torch.equal(painted, expected_painted)  # "Exactly the three survivors' pixels must be painted.", reporting both painted counts and the painted pixels
│   └── for culled_index in (3, 4)
│       └── assert not (seg_map == pc_data.labels[culled_index]).any()  # "No pixel may carry the label of a point that culled out.", reporting culled_index, its label and the pixels carrying it
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
│   ├── calls PointCloud(xyz=three float32 points at depths one, two and four projecting to three separate pixels, data=one distinguishable normal per point)  # -> pc_data
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)  # -> camera
│   ├── impls resolution = (100, 100)
│   ├── calls render_normal_from_point_cloud_3d(pc=pc_data, camera=camera, resolution=resolution)  # -> normal_map
│   ├── impls camera_normals = pc_data.normals * [1.0, -1.0, -1.0] as a float32 tensor  # the camera frame the renderer rotates a normal into: the identity OpenGL pose reaches OpenCV as the world-to-camera rotation diag(1, -1, -1)
│   ├── impls pixels = [(50, 50), (43, 62), (56, 37)]  # the pixels the three points project to, in point order
│   └── for point_index, (row, column) in enumerate(pixels)
│       └── assert (normal_map[:, row, column] == camera_normals[point_index]).all()  # "Each rendered pixel must carry the normal of the point that projected onto it.", reporting point_index, row, column and both normals; the normal of the point that projected there, not another point's
├── def test_render_normal_ignores_the_points_that_culled_out() -> None
│   ├── # A cloud with several survivors and several culled points renders only the survivors, the regime a fixture of one or two in-bounds points cannot reach.
│   ├── calls PointCloud(xyz=three float32 points inside the image bounds and two placed far outside them, data=one distinguishable normal per point)  # -> pc_data
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)  # -> camera
│   ├── impls resolution = (100, 100)
│   ├── calls render_normal_from_point_cloud_3d(pc=pc_data, camera=camera, resolution=resolution)  # -> normal_map
│   ├── impls painted = the pixels where any channel of normal_map differs from 0.0  # 0.0 is the renderer's default ignore_value
│   ├── impls expected_painted = a bool map of shape resolution filled with False
│   ├── for row, column in [(50, 50), (43, 62), (56, 37)]
│   │   └── impls expected_painted[row, column] = True
│   ├── assert torch.equal(painted, expected_painted)  # "Exactly the three survivors' pixels must be painted.", reporting both painted counts and the painted pixels
│   ├── impls camera_normals = pc_data.normals * [1.0, -1.0, -1.0] as a float32 tensor  # the camera frame the renderer rotates a normal into: the identity OpenGL pose reaches OpenCV as the world-to-camera rotation diag(1, -1, -1)
│   └── for culled_index in (3, 4)
│       ├── impls culled_normal = camera_normals[culled_index] reshaped to [3, 1, 1]
│       └── assert not (normal_map == culled_normal).all(dim=0).any()  # "No pixel may carry the normal of a point that culled out.", reporting culled_index, its camera-frame normal and the pixels carrying it
├── def test_render_normal_takes_the_nearest_point_where_two_share_a_pixel() -> None
│   ├── # Two points on one ray paint the nearer one's normal, so the attribute follows the same occlusion the depth map resolves.
│   ├── calls PointCloud(xyz=two float32 points on one ray at depths three and one, data=one distinguishable normal per point)  # -> pc_data
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)  # -> camera
│   ├── impls resolution = (100, 100)
│   ├── calls render_normal_from_point_cloud_3d(pc=pc_data, camera=camera, resolution=resolution)  # -> normal_map
│   ├── impls camera_normals = pc_data.normals * [1.0, -1.0, -1.0] as a float32 tensor  # the camera frame the renderer rotates a normal into: the identity OpenGL pose reaches OpenCV as the world-to-camera rotation diag(1, -1, -1)
│   └── assert (normal_map[:, 50, 50] == camera_normals[1]).all()  # "The pixel two points share must carry the near point's normal.", reporting that pixel's normal and both points' camera-frame normals
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
│   └── for point_size in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
│       ├── calls create_circular_kernel_offsets(point_size=point_size, device=torch.device('cpu'))  # -> kernel_offsets
│       ├── impls offsets = an empty set
│       ├── for each offset of kernel_offsets
│       │   └── impls offsets gains offset's (y, x) as an integer pair
│       ├── impls unmatched = an empty set  # the offsets whose negation is missing from the same set
│       ├── for each (y, x) of offsets
│       │   └── if (-y, -x) is not in offsets
│       │       └── impls unmatched gains (y, x)
│       └── assert not unmatched  # "Every kernel offset must have its negation in the kernel, otherwise the disc reaches farther on one side of the point than on the other.", reporting point_size and the sorted unmatched and offsets
├── def test_create_circular_kernel_offsets_membership_is_the_radius_rule() -> None
│   ├── # The kernel is exactly the cells whose centre lies inside the disc, neither more nor fewer, checked against a radius rule the test derives itself.
│   └── for point_size in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
│       ├── calls create_circular_kernel_offsets(point_size=point_size, device=torch.device('cpu'))  # -> kernel_offsets
│       ├── impls offsets = an empty set
│       ├── for each offset of kernel_offsets
│       │   └── impls offsets gains offset's (y, x) as an integer pair
│       ├── impls kernel_radius = point_size / 2.0
│       ├── impls search_reach = math.ceil(point_size) + 1  # a generous search box, one cell past the disc
│       ├── impls expected = an empty set  # the integer cells of the search box whose distance from the origin is within kernel_radius
│       ├── for each y from -search_reach to search_reach
│       │   └── for each x from -search_reach to search_reach
│       │       └── if the distance of (y, x) from the origin is at most kernel_radius
│       │           └── impls expected gains (y, x)
│       └── assert offsets == expected and len(offsets) == kernel_offsets.shape[0]  # "The kernel must hold exactly the cells whose centre lies inside the disc, and must not repeat a cell, otherwise a disc pixel is dilated twice.", reporting point_size, kernel_radius, both set differences, len(offsets) and kernel_offsets.shape
├── def test_create_circular_kernel_offsets_dilates_a_point_into_a_centred_disc() -> None
│   ├── # One rendered point grows into a disc centred on its own pixel, which is the kernel's symmetry seen through the renderer that uses it.
│   ├── impls expected_covered_counts = {1.0: 1, 1.5: 1, 2.0: 5, 3.0: 9, 4.0: 13, 5.0: 21}  # the pixel count each point size's disc holds
│   ├── impls principal_point = 20.5
│   ├── calls PointCloud(xyz=one float32 point on the optical axis at depth one)  # -> pc_data
│   ├── calls _build_camera(focal=100.0, principal_point=principal_point)  # -> camera
│   ├── impls resolution = (41, 41)
│   ├── impls center_pixel = math.floor(principal_point)  # the point sits on the optical axis, so it lands on the pixel holding the principal point, the same pixel index on both axes
│   └── for point_size in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
│       ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=resolution, return_mask=True, point_size=point_size)  # -> the depth map, discarded, and valid_mask
│       ├── impls covered = an empty set  # the covered pixels as offsets from the pixel the point itself landed on
│       ├── for each pixel valid_mask marks
│       │   └── impls covered gains (the pixel's row - center_pixel, the pixel's column - center_pixel)
│       ├── impls unmatched = an empty set
│       ├── for each (y, x) of covered
│       │   └── if (-y, -x) is not in covered
│       │       └── impls unmatched gains (y, x)
│       ├── assert not unmatched  # "The covered pixels must be symmetric about the point's own pixel.", reporting point_size, center_pixel and the sorted unmatched and covered
│       └── assert len(covered) == expected_covered_counts[point_size]  # "A point must dilate into the disc of pixels its point size reaches.", reporting point_size, len(covered), the expected count and the sorted covered
└── def _build_camera(focal: float, principal_point: float) -> Camera
    ├── # Builds the identity-pose OpenGL pinhole camera on the CPU whose own extents match the requested resolution, so no intrinsics rescaling moves the point off centre.
    ├── calls build_camera_intrinsics(model='pinhole', params=the shared focal and principal point with the extents twice that point implies, intr_convention='standard', device=torch.device('cpu'))
    ├── calls CameraExtrinsics(extrinsics=a float32 [4, 4] identity, extr_convention='opengl', device=torch.device('cpu'))
    ├── calls Camera(intrinsics=the intrinsics it built, extrinsics=the extrinsics it built, device=torch.device('cpu'))
    └── return  # that camera
```
