# `models/three_d/point_cloud/render/` tests skeleton

## Tests implementation structure

`tests/models/three_d/point_cloud/render/test_render_rgb.py`

```text
test_render_rgb.py
├── import pytest
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.render import render_rgb_from_point_cloud
├── def test_render_rgb_basic
│   ├── # Four coloured points at distinct depths render to a [3, H, W] float32 image whose values stay inside [0, 1].
│   ├── calls PointCloud(xyz=four points at distinct depths, data={'rgb': their four colours})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=False)
│   └── impls assert the image is [3, 100, 100] float32 with values in [0, 1]
├── def test_render_rgb_with_mask
│   ├── # return_mask pairs the image with a bool [H, W] coverage mask that is neither empty nor full, and the uncovered pixels stay at the ignore value.
│   ├── calls PointCloud(xyz=two points at distinct depths, data={'rgb': their two colours})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=True)
│   ├── impls assert the mask is [100, 100] bool with a partial, non-zero count
│   └── impls assert the image is 0.0 everywhere the mask is False
├── def test_render_rgb_color_normalization
│   ├── # Colours given in the 0-255 range are normalized to [0, 1] rather than clipped, so an unnormalized point cloud renders the same as a normalized one.
│   ├── calls PointCloud(xyz=two points at distinct depths, data={'rgb': colours in the 0-255 range})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=True)
│   └── impls assert the image stays within [0, 1] and the mask covers something
├── def test_render_rgb_depth_sorting
│   ├── # Two points on one pixel resolve to the nearer point's colour, the back-to-front ordering the rasterizer depends on.
│   ├── calls PointCloud(xyz=two points on the same pixel at different depths, data={'rgb': a far and a near colour})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100))
│   └── impls assert the contested pixel carries the nearer point's colour
├── def test_render_rgb_points_behind_camera
│   ├── # A point behind the camera is culled instead of wrapping into the image, leaving only the in-front point covered.
│   ├── calls PointCloud(xyz=one point behind and one in front of the camera, data={'rgb': their two colours})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=True)
│   └── impls assert the mask covers at least the in-front point and the image stays non-negative
├── def test_render_rgb_custom_ignore_value
│   ├── # A caller-supplied ignore_value fills every pixel no point projected onto.
│   ├── calls PointCloud(xyz=one point, data={'rgb': one colour})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), ignore_value=ignore_value)
│   └── impls assert the background pixels all carry ignore_value across every channel
├── def test_render_rgb_missing_rgb_field
│   ├── # A point cloud with no rgb field is rejected rather than rendered as a default colour.
│   ├── calls PointCloud(xyz=one point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   └── with pytest.raises(AssertionError)
│       └── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100))
├── def test_render_rgb_invalid_inputs
│   ├── # The rejected-input surface: a non-PointCloud pc, a non-CameraIntrinsics intrinsics, a malformed extrinsics matrix, and a non-positive resolution.
│   ├── calls PointCloud(xyz=one point, data={'rgb': one colour})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls render_rgb_from_point_cloud(pc=None, camera=camera, resolution=(100, 100))
│   ├── with pytest.raises(AssertionError)
│   │   └── calls CameraExtrinsics(extrinsics=a [3, 3] matrix, extr_convention="opengl", device=cpu)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls Camera(intrinsics=a raw [4, 4] tensor, extrinsics=a valid CameraExtrinsics, device=cpu)
│   └── with pytest.raises(AssertionError)
│       └── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(0, 100))
└── def _build_camera(focal: float, principal_point: float) -> Camera
    ├── # Builds the identity-pose opengl pinhole camera on the cpu that every test in this file renders through.
    ├── calls build_camera_intrinsics(model="pinhole", params=the fx/fy/cx/cy/h/w built from focal and principal_point, intr_convention="standard", device=cpu)
    ├── calls CameraExtrinsics(extrinsics=the identity [4, 4], extr_convention="opengl", device=cpu)
    └── calls Camera(intrinsics=intrinsics, extrinsics=extrinsics, device=cpu)
```

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
├── def test_render_depth_basic
│   ├── # Four points at distinct depths render to a [H, W] float32 map whose covered pixels all carry positive depth.
│   ├── calls PointCloud(xyz=four points at distinct depths)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=False)
│   └── impls assert the map is [100, 100] float32 and every non-background depth is positive
├── def test_render_depth_with_mask
│   ├── # return_mask pairs the map with a bool [H, W] coverage mask that is neither empty nor full, and the uncovered pixels stay at the default -1.0 ignore value.
│   ├── calls PointCloud(xyz=three points at distinct depths)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=True)
│   ├── impls assert the mask is [100, 100] bool with a partial, non-zero count
│   └── impls assert the masked-in depths are positive and the masked-out depths are -1.0
├── def test_render_depth_sorting
│   ├── # Two points on one pixel resolve to the nearer point's depth, the back-to-front ordering the rasterizer depends on.
│   ├── calls PointCloud(xyz=two points on the same pixel at different depths)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=resolution)
│   └── impls assert the contested pixel carries the nearer depth
├── def test_render_depth_custom_ignore_value
│   ├── # A caller-supplied ignore_value fills every pixel no point projected onto.
│   ├── calls PointCloud(xyz=one point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), ignore_value=custom_ignore)
│   └── impls assert the overwhelming majority of pixels carry custom_ignore
├── def test_render_depth_points_behind_camera
│   ├── # A point behind the camera is culled instead of wrapping into the image, leaving only the in-front point covered.
│   ├── calls PointCloud(xyz=one point behind and one in front of the camera)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=True)
│   └── impls assert the mask covers at least the in-front point and every covered depth is positive
├── def test_render_depth_multiple_points_per_pixel
│   ├── # A long focal length collapsing four points onto one pixel still renders one finite nearest depth rather than a conflict.
│   ├── calls PointCloud(xyz=four near-coincident points at different depths)
│   ├── calls _build_camera(focal=1000.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100))
│   └── impls assert the shared pixel carries the nearest of the four depths
├── def test_render_depth_intrinsics_scaling
│   ├── # The same point cloud and camera render at a smaller and a larger resolution, so the intrinsics are scaled to the requested resolution rather than fixed at their native one.
│   ├── calls PointCloud(xyz=two points at distinct depths)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(50, 50))
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(200, 200))
│   └── impls assert each map matches its requested resolution and carries rendered content
├── def test_render_depth_invalid_inputs
│   ├── # The rejected-input surface: a non-PointCloud pc, a non-CameraIntrinsics intrinsics, a malformed extrinsics matrix, and a non-positive resolution.
│   ├── calls PointCloud(xyz=one point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls render_depth_from_point_cloud(pc="not a point cloud", camera=valid_camera, resolution=(100, 100))
│   ├── with pytest.raises(AssertionError)
│   │   └── calls Camera(intrinsics=a raw [4, 4] tensor, extrinsics=a valid CameraExtrinsics, device=cpu)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls CameraExtrinsics(extrinsics=a [3, 3] matrix, extr_convention="opengl", device=cpu)
│   └── with pytest.raises(AssertionError)
│       └── calls render_depth_from_point_cloud(pc=valid_pc_data, camera=valid_camera, resolution=(0, 100))
└── def _build_camera(focal: float, principal_point: float) -> Camera
    ├── # Builds the identity-pose opengl pinhole camera on the cpu that every test in this file renders through.
    ├── calls build_camera_intrinsics(model="pinhole", params=the fx/fy/cx/cy/h/w built from focal and principal_point, intr_convention="standard", device=cpu)
    ├── calls CameraExtrinsics(extrinsics=the identity [4, 4], extr_convention="opengl", device=cpu)
    └── calls Camera(intrinsics=intrinsics, extrinsics=extrinsics, device=cpu)
```

`tests/models/three_d/point_cloud/render/test_render_segmentation.py`

```text
test_render_segmentation.py
├── import pytest
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from models.three_d.point_cloud.render import render_segmentation_from_point_cloud
├── def test_render_segmentation_basic
│   ├── # Three labelled points render to an [H, W] int64 label map whose uncovered pixels carry the default 255 ignore label.
│   ├── calls PointCloud(xyz=three points at distinct depths, data={'labels': their three labels})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_segmentation_from_point_cloud(pc=pc_data, key="labels", camera=camera, resolution=(100, 100), return_mask=False)
│   └── impls assert the map is [100, 100] int64 and some pixels carry 255
├── def test_render_segmentation_with_mask
│   ├── # return_mask pairs the label map with a bool [H, W] coverage mask, and every covered pixel carries a real label rather than the ignore label.
│   ├── calls PointCloud(xyz=two points at distinct depths, data={'labels': their two labels})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_segmentation_from_point_cloud(pc=pc_data, key="labels", camera=camera, resolution=(100, 100), return_mask=True)
│   ├── impls assert the mask is [100, 100] bool with a non-zero count
│   └── impls assert no masked-in pixel carries 255
├── def test_render_segmentation_depth_sorting
│   ├── # Two points on one pixel resolve to the nearer point's label, and the map stays int64 rather than being cast by the dilation path.
│   ├── calls PointCloud(xyz=two points on the same pixel at different depths, data={'labels': a far and a near label})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_segmentation_from_point_cloud(pc=pc_data, key="labels", camera=camera, resolution=(100, 100))
│   └── impls assert the map is int64 and the contested pixel carries the nearer point's label
├── def test_render_segmentation_points_behind_camera
│   ├── # A point behind the camera is culled instead of wrapping into the image, leaving only the in-front point labelled.
│   ├── calls PointCloud(xyz=one point behind and one in front of the camera, data={'labels': their two labels})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_segmentation_from_point_cloud(pc=pc_data, key="labels", camera=camera, resolution=(100, 100), return_mask=True)
│   └── impls assert the mask covers at least the in-front point and no covered pixel carries 255
├── def test_render_segmentation_custom_ignore_value
│   ├── # A caller-supplied ignore_value replaces 255 as the label filling every pixel no point projected onto.
│   ├── calls PointCloud(xyz=one point, data={'labels': one label})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_segmentation_from_point_cloud(pc=pc_data, key="labels", camera=camera, resolution=(100, 100), ignore_value=ignore_value)
│   └── impls assert the background pixels carry ignore_value
├── def test_render_segmentation_missing_labels
│   ├── # A point cloud with no field under the requested key is rejected rather than rendered as all-ignore.
│   ├── calls PointCloud(xyz=one point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   └── with pytest.raises(AssertionError)
│       └── calls render_segmentation_from_point_cloud(pc=pc_data, key="labels", camera=camera, resolution=(100, 100))
├── def test_render_segmentation_invalid_inputs
│   ├── # The rejected-input surface: a non-PointCloud pc, a non-CameraIntrinsics intrinsics, a malformed extrinsics matrix, and a non-positive resolution.
│   ├── calls PointCloud(xyz=one point, data={'labels': one label})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls render_segmentation_from_point_cloud(pc=None, key="labels", camera=valid_camera, resolution=(100, 100))
│   ├── with pytest.raises(AssertionError)
│   │   └── calls Camera(intrinsics=a raw [4, 4] tensor, extrinsics=a valid CameraExtrinsics, device=cpu)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls CameraExtrinsics(extrinsics=a [3, 3] matrix, extr_convention="opengl", device=cpu)
│   └── with pytest.raises(AssertionError)
│       └── calls render_segmentation_from_point_cloud(pc=pc_data, key="labels", camera=valid_camera, resolution=(0, 100))
└── def _build_camera(focal: float, principal_point: float) -> Camera
    ├── # Builds the identity-pose opengl pinhole camera on the cpu that every test in this file renders through.
    ├── calls build_camera_intrinsics(model="pinhole", params=the fx/fy/cx/cy/h/w built from focal and principal_point, intr_convention="standard", device=cpu)
    ├── calls CameraExtrinsics(extrinsics=the identity [4, 4], extr_convention="opengl", device=cpu)
    └── calls Camera(intrinsics=intrinsics, extrinsics=extrinsics, device=cpu)
```
