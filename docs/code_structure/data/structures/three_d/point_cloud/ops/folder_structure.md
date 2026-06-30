# data/structures/three_d/point_cloud/ops — folder structure

## Code folder structure

```text
data/structures/three_d/point_cloud/ops/
├── __init__.py
├── apply_transform.py            # applies a 4x4 transform to points in homogeneous coordinates
├── correspondences.py
├── generate_change_map.py
├── grid_sampling.py
├── normalization.py
├── world_to_camera_transform.py   # high-level world-to-camera point transform, implemented via apply_transform
├── knn/
│   ├── __init__.py
│   ├── knn.py
│   ├── knn_faiss.py
│   ├── knn_pytorch3d.py
│   ├── knn_scipy.py
│   └── knn_torch.py
├── lift/
│   └── lift.py
├── rendering/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── apply_point_size_postprocessing.py
│   │   ├── create_circular_kernel_offsets.py
│   │   ├── prepare_points_for_rendering.py    # transforms (via world_to_camera_transform) + projects points to pixels with frustum cull, delegating the max_divide/num_divide OOM-chunking to chunked_matmul
│   │   └── validate_rendering_inputs.py
│   ├── render_depth.py
│   ├── render_mask.py
│   ├── render_normal.py
│   ├── render_rgb.py
│   ├── render_rgb_o3d.py
│   ├── render_rgb_volumetric.py
│   └── render_segmentation.py
├── sampling/
│   ├── __init__.py
│   ├── cylinder_sampling.py
│   ├── grid_sampling_3d.py
│   └── grid_sampling_3d_v2.py
└── set_ops/
    ├── __init__.py
    ├── intersection.py
    └── symmetric_difference.py
```

## Tests folder structure

```text
tests/data/structures/three_d/point_cloud/
└── test_select_random_select.py
```
