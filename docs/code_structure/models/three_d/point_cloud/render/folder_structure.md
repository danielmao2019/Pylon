# `models/three_d/point_cloud/render/` folder skeleton

## Code folder structure

```text
models/three_d/point_cloud/render/
├── __init__.py  # MODELS.THREE_D.POINT_CLOUD.RENDER package API surface.
├── common/
│   ├── __init__.py
│   ├── apply_point_size_postprocessing.py
│   ├── create_circular_kernel_offsets.py
│   ├── prepare_points_for_rendering.py  # world-to-camera via world_to_camera_transform, camera-to-image via CameraIntrinsics.project, frustum cull, OOM-adaptive batching
│   └── validate_rendering_inputs.py
├── render_depth.py
├── render_mask.py
├── render_normal.py
├── render_rgb.py
├── render_rgb_o3d.py
├── render_rgb_volumetric.py
└── render_segmentation.py
```

## Tests folder structure

```text
tests/models/three_d/point_cloud/render/
├── test_create_circular_kernel_offsets.py  # the dilation kernel: that its disc is centred on the point and holds exactly the cells inside the radius
├── test_render_depth.py  # depth rendering: resolution, the valid mask, occlusion order, and the inputs it refuses
├── test_render_normal.py  # normal rendering: the normal each pixel's owning point carries, and the culled points it must not paint
├── test_render_rgb.py  # rgb rendering: the colour each pixel's owning point carries, and the culled points it must not paint
└── test_render_segmentation.py  # segmentation rendering: the label each pixel's owning point carries, and the culled points it must not paint
```
