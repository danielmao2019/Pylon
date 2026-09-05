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
│   ├── # Four coloured points at distinct depths render to an image whose shape, dtype and value range are checked.
│   ├── calls PointCloud(xyz=four points at distinct depths, data={'rgb': their four colours})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=False)
│   ├── impls assert the image shape is (3, 100, 100) and its dtype is torch.float32
│   └── impls assert the image min is at least 0.0 and its max is at most 1.0
├── def test_render_rgb_with_mask
│   ├── # return_mask pairs the image with a bool mask whose coverage is partial and non-empty, and the uncovered pixels are checked against 0.0.
│   ├── calls PointCloud(xyz=two points at distinct depths, data={'rgb': their two colours})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=True)
│   ├── impls assert the image shape is (3, 100, 100), the mask shape is (100, 100), and the mask dtype is torch.bool
│   ├── impls assert the mask count is above 0 and below 100 * 100
│   └── impls assert the image is 0.0 at every channel of every pixel the mask excludes
├── def test_render_rgb_color_normalization
│   ├── # Colours given in the 0-255 range are accepted, and the render is checked for landing inside [0, 1] with the mask covering something.
│   ├── calls PointCloud(xyz=two points at distinct depths, data={'rgb': colours in the 0-255 range})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=True)
│   ├── impls assert the image max is at most 1.0 and its min is at least 0.0
│   └── impls assert the mask covers at least one pixel
├── def test_render_rgb_depth_sorting
│   ├── # Two coincident points at different depths render, and the assertion checks only that some colour was written.
│   ├── calls PointCloud(xyz=two points on the same pixel at different depths, data={'rgb': a far and a near colour})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100))
│   └── impls assert some element of the image is greater than 0.0
├── def test_render_rgb_points_behind_camera
│   ├── # A point behind and a point in front of the camera render together, and the mask is checked for covering at least one pixel.
│   ├── calls PointCloud(xyz=one point behind and one in front of the camera, data={'rgb': their two colours})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=True)
│   ├── impls assert the mask count is at least 1
│   └── impls assert the image is at least 0.0 at every covered pixel
├── def test_render_rgb_custom_ignore_value
│   ├── # A caller-supplied ignore_value of -1.0 is checked on the pixels whose channel 0 already carries it.
│   ├── calls PointCloud(xyz=one point, data={'rgb': one colour})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), ignore_value=ignore_value)
│   └── impls assert every channel equals ignore_value at the pixels whose channel 0 equals ignore_value
├── def test_render_rgb_missing_rgb_field
│   ├── # A point cloud with no rgb field is rejected rather than rendered as a default colour.
│   ├── calls PointCloud(xyz=one point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   └── with pytest.raises(AssertionError)
│       └── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100))
├── def test_render_rgb_invalid_inputs
│   ├── # The rejected-input surface: a non-PointCloud pc, a malformed extrinsics matrix, a non-CameraIntrinsics intrinsics, and a non-positive resolution.
│   ├── calls PointCloud(xyz=one point, data={'rgb': one colour})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls render_rgb_from_point_cloud(pc=None, camera=camera, resolution=(100, 100))
│   ├── with pytest.raises(AssertionError)
│   │   └── calls CameraExtrinsics(extrinsics=a [3, 3] matrix, extr_convention="opengl", device=cpu)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls Camera(intrinsics=a raw [4, 4] tensor, extrinsics=CameraExtrinsics(extrinsics=the identity [4, 4], extr_convention="opengl", device=cpu), device=cpu)
│   └── with pytest.raises(AssertionError)
│       └── calls render_rgb_from_point_cloud(pc=pc_data, camera=camera, resolution=(0, 100))
└── def _build_camera(focal: float, principal_point: float) -> Camera
    ├── # Builds the identity-pose opengl pinhole camera on the cpu that every test in this file renders through.
    ├── calls build_camera_intrinsics(model="pinhole", params=the fx/fy/cx/cy/h/w built from focal and principal_point, intr_convention="standard", device=cpu)
    ├── calls CameraExtrinsics(extrinsics=the identity [4, 4], extr_convention="opengl", device=cpu)
    ├── calls Camera(intrinsics=intrinsics, extrinsics=extrinsics, device=cpu)
    └── return  # the identity-pose opengl pinhole Camera
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
│   ├── # Four points at distinct depths render to a map whose shape and dtype are checked, and whose non-background depths are checked for being positive.
│   ├── calls PointCloud(xyz=four points at distinct depths)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=resolution, return_mask=False)
│   ├── impls assert the map shape is (100, 100) and its dtype is torch.float32
│   ├── impls valid_depths = the map's entries that are not the -1.0 background
│   └── impls assert every entry of valid_depths is greater than 0
├── def test_render_depth_with_mask
│   ├── # return_mask pairs the map with a bool mask whose coverage is partial and non-empty, and both sides of the mask are checked against the -1.0 default.
│   ├── calls PointCloud(xyz=three points at distinct depths)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=resolution, return_mask=True)
│   ├── impls assert both the map and the mask are (100, 100), and the mask dtype is torch.bool
│   ├── impls assert the mask count is above 0 and below 100 * 100
│   └── impls assert the covered depths are positive and the uncovered depths are -1.0
├── def test_render_depth_sorting
│   ├── # Two coincident points at depths 1 and 3 render, and any surviving depth is checked for falling under 1.5.
│   ├── calls PointCloud(xyz=two points on the same pixel at depths 1.0 and 3.0)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=resolution)
│   ├── impls valid_depths = the map's entries that are not the -1.0 background
│   └── if len(valid_depths) > 0
│       └── impls assert the minimum of valid_depths is under 1.5
├── def test_render_depth_custom_ignore_value
│   ├── # A single point rendered with a custom_ignore of -999.0 leaves that value on over 90% of the image.
│   ├── calls PointCloud(xyz=one point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), ignore_value=custom_ignore)
│   └── impls assert the count of pixels equal to custom_ignore exceeds 0.9 * 100 * 100
├── def test_render_depth_points_behind_camera
│   ├── # A point behind and a point in front of the camera render together, and the mask is checked for covering at least one pixel.
│   ├── calls PointCloud(xyz=one point behind and one in front of the camera)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100), return_mask=True)
│   ├── impls assert the mask count is at least 1
│   └── impls assert every covered depth is greater than 0
├── def test_render_depth_multiple_points_per_pixel
│   ├── # Four near-coincident points render at a long focal length, and any surviving depth in the sampled centre region is checked for falling under 1.5.
│   ├── calls PointCloud(xyz=four near-coincident points at different depths)
│   ├── calls _build_camera(focal=1000.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(100, 100))
│   ├── impls valid_depths = the non-background entries of the [48:52, 48:52] centre region
│   └── if len(valid_depths) > 0
│       └── impls assert the minimum of valid_depths is under 1.5
├── def test_render_depth_intrinsics_scaling
│   ├── # The same point cloud and camera render at two resolutions, and each map is checked for matching its requested shape and carrying content.
│   ├── calls PointCloud(xyz=two points at distinct depths)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(50, 50))
│   ├── calls render_depth_from_point_cloud(pc=pc_data, camera=camera, resolution=(200, 200))
│   ├── impls assert the two map shapes are (50, 50) and (200, 200)
│   └── impls assert each map holds at least one entry that is not -1.0
├── def test_render_depth_invalid_inputs
│   ├── # The rejected-input surface: a non-PointCloud pc, a non-CameraIntrinsics intrinsics, a malformed extrinsics matrix, and a non-positive resolution.
│   ├── calls PointCloud(xyz=one point)
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls render_depth_from_point_cloud(pc="not a point cloud", camera=valid_camera, resolution=(100, 100))
│   ├── with pytest.raises(AssertionError)
│   │   └── calls Camera(intrinsics=a raw [4, 4] tensor, extrinsics=CameraExtrinsics(extrinsics=the identity [4, 4], extr_convention="opengl", device=cpu), device=cpu)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls CameraExtrinsics(extrinsics=a [3, 3] matrix, extr_convention="opengl", device=cpu)
│   └── with pytest.raises(AssertionError)
│       └── calls render_depth_from_point_cloud(pc=valid_pc_data, camera=valid_camera, resolution=(0, 100))
└── def _build_camera(focal: float, principal_point: float) -> Camera
    ├── # Builds the identity-pose opengl pinhole camera on the cpu that every test in this file renders through.
    ├── calls build_camera_intrinsics(model="pinhole", params=the fx/fy/cx/cy/h/w built from focal and principal_point, intr_convention="standard", device=cpu)
    ├── calls CameraExtrinsics(extrinsics=the identity [4, 4], extr_convention="opengl", device=cpu)
    ├── calls Camera(intrinsics=intrinsics, extrinsics=extrinsics, device=cpu)
    └── return  # the identity-pose opengl pinhole Camera
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
│   ├── # Three labelled points render to a map whose shape and int64 dtype are checked, and which is checked for carrying the 255 default somewhere.
│   ├── calls PointCloud(xyz=three points at distinct depths, data={'labels': their three labels})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_segmentation_from_point_cloud(pc=pc_data, key="labels", camera=camera, resolution=(100, 100), return_mask=False)
│   ├── impls assert the map shape is (100, 100) and its dtype is torch.int64
│   └── impls assert the count of pixels equal to 255 is above 0
├── def test_render_segmentation_with_mask
│   ├── # return_mask pairs the label map with a bool mask that covers something, and every covered pixel is checked for carrying a label other than 255.
│   ├── calls PointCloud(xyz=two points at distinct depths, data={'labels': their two labels})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_segmentation_from_point_cloud(pc=pc_data, key="labels", camera=camera, resolution=(100, 100), return_mask=True)
│   ├── impls assert both the map and the mask are (100, 100), and the mask dtype is torch.bool
│   ├── impls assert the mask count is above 0
│   └── impls assert no covered pixel carries 255
├── def test_render_segmentation_depth_sorting
│   ├── # Two coincident points at different depths render, and the assertion checks only that the label map is int64.
│   ├── calls PointCloud(xyz=two points on the same pixel at different depths, data={'labels': a far and a near label})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_segmentation_from_point_cloud(pc=pc_data, key="labels", camera=camera, resolution=(100, 100))
│   └── impls assert the map dtype is torch.int64
├── def test_render_segmentation_points_behind_camera
│   ├── # A point behind and a point in front of the camera render together, and the mask is checked for covering at least one pixel.
│   ├── calls PointCloud(xyz=one point behind and one in front of the camera, data={'labels': their two labels})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_segmentation_from_point_cloud(pc=pc_data, key="labels", camera=camera, resolution=(100, 100), return_mask=True)
│   ├── impls assert the mask count is at least 1
│   └── impls assert no covered pixel carries 255
├── def test_render_segmentation_custom_ignore_value
│   ├── # A single labelled point renders with a custom ignore_value of -1, and the count of pixels equal to ignore_value is checked for being above 0.
│   ├── calls PointCloud(xyz=one point, data={'labels': one label})
│   ├── calls _build_camera(focal=100.0, principal_point=50.0)
│   ├── calls render_segmentation_from_point_cloud(pc=pc_data, key="labels", camera=camera, resolution=(100, 100), ignore_value=ignore_value)
│   └── impls assert the count of pixels equal to ignore_value is above 0
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
│   │   └── calls Camera(intrinsics=a raw [4, 4] tensor, extrinsics=CameraExtrinsics(extrinsics=the identity [4, 4], extr_convention="opengl", device=cpu), device=cpu)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls CameraExtrinsics(extrinsics=a [3, 3] matrix, extr_convention="opengl", device=cpu)
│   └── with pytest.raises(AssertionError)
│       └── calls render_segmentation_from_point_cloud(pc=pc_data, key="labels", camera=valid_camera, resolution=(0, 100))
└── def _build_camera(focal: float, principal_point: float) -> Camera
    ├── # Builds the identity-pose opengl pinhole camera on the cpu that every test in this file renders through.
    ├── calls build_camera_intrinsics(model="pinhole", params=the fx/fy/cx/cy/h/w built from focal and principal_point, intr_convention="standard", device=cpu)
    ├── calls CameraExtrinsics(extrinsics=the identity [4, 4], extr_convention="opengl", device=cpu)
    ├── calls Camera(intrinsics=intrinsics, extrinsics=extrinsics, device=cpu)
    └── return  # the identity-pose opengl pinhole Camera
```
