# `models/three_d/point_cloud/ops/` folder skeleton

## Code folder structure

```text
ops/
├── __init__.py                     # MODELS.THREE_D.POINT_CLOUD.OPS package API surface.
├── apply_transform.py              # applies a 4x4 transform to points in homogeneous coordinates via chunked_matmul
├── correspondences.py
├── generate_change_map.py
├── grid_sampling.py
├── normalization.py
├── world_to_camera_transform.py    # high-level world-to-camera point transform, implemented via apply_transform
├── knn/
│   ├── __init__.py
│   ├── knn.py
│   ├── knn_faiss.py
│   ├── knn_pytorch3d.py
│   ├── knn_scipy.py
│   └── knn_torch.py
├── lift/
│   └── lift.py
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
tests/models/three_d/point_cloud/ops/
```
