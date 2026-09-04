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
├── display.py  # Scene-model display rendering with snapshot caching and camera overlays.
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
├── test_render_depth.py         # render_depth_from_point_cloud end to end: map shape/dtype, mask pairing, depth ordering, ignore_value, behind-camera culling, intrinsics scaling, rejected inputs
├── test_render_rgb.py           # render_rgb_from_point_cloud end to end: image shape/dtype/range, mask pairing, 0-255 colour normalization, depth ordering, behind-camera culling, ignore_value, missing rgb, rejected inputs
└── test_render_segmentation.py  # render_segmentation_from_point_cloud end to end: label map shape/dtype, mask pairing, depth ordering, behind-camera culling, ignore_value, missing label field, rejected inputs
```
