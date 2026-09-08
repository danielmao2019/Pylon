# Vision-3D Transforms Tests Structure

## Tests implementation structure

`tests/data/transforms/vision_3d/test_random_rigid_transform.py`

```text
test_random_rigid_transform.py
├── import numpy as np
├── import pytest
├── import torch
├── from data.structures.three_d.camera.extrinsics.rotation.rodrigues import rodrigues_to_matrix
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.transforms.vision_3d.random_rigid_transform import RandomRigidTransform
├── from models.three_d.point_cloud.ops import apply_transform
├── DEVICE  # torch.device('cuda') when one is available, otherwise torch.device('cpu') — the device every cloud and transform in this suite is built on
├── def test_random_rigid_transform()
│   ├── # The posed triplet stays consistent: the returned transform still carries the returned source onto the returned target, the target is untouched, the source moved, and both feature fields survived.
│   ├── calls create_random_point_cloud(1000)
│   ├── calls create_point_cloud(src_points)
│   ├── impls original_src_xyz = a copy of that cloud's coordinates
│   ├── impls original_src_feat = a copy of its feature field
│   ├── calls create_random_transform()
│   ├── calls apply_transform(src_points, transform)
│   ├── calls create_point_cloud(tgt_points)
│   ├── impls original_tgt_xyz = a copy of that cloud's coordinates
│   ├── impls original_tgt_feat = a copy of its feature field
│   ├── calls RandomRigidTransform(rot_mag=45.0, trans_mag=0.5)
│   ├── impls new_src_pc, new_tgt_pc, new_transform = what it returns for the pair at seed 42
│   ├── calls apply_transform(new_src_pc.xyz, new_transform)
│   ├── assert what that produces matches new_tgt_pc.xyz
│   ├── assert new_tgt_pc.xyz still matches original_tgt_xyz
│   ├── assert new_src_pc.xyz differs from original_src_xyz
│   └── assert both feature fields still match the copies taken before the call
├── def test_random_rigid_transform_deterministic()
│   ├── # One seed reproduces one pose, which is what lets a dataset replay a datapoint across epochs.
│   ├── impls torch seeded at 42
│   ├── impls numpy seeded at 42
│   ├── calls create_random_point_cloud(1000)
│   ├── calls create_point_cloud(src_points)
│   ├── calls create_random_transform()
│   ├── calls apply_transform(src_points, transform)
│   ├── calls create_point_cloud(tgt_points)
│   ├── calls RandomRigidTransform(rot_mag=45.0, trans_mag=0.5)
│   ├── impls the first triplet = what it returns for freshly built copies of the pair at seed 42
│   ├── impls the second triplet = what it returns for another pair of copies at that same seed
│   ├── assert the two posed sources match
│   └── assert the two adjusted transforms match
├── def create_random_point_cloud(num_points=1000)
│   ├── # Draws the float32 [N, 3] coordinate block the cases here pose.
│   ├── impls points = num_points rows of three standard-normal values, as float32 on DEVICE
│   └── return points
├── def create_random_transform()
│   ├── # Builds the float32 [4, 4] ground-truth transform a pair is constructed around.
│   ├── impls angle_torch = a random angle over the full turn, as float32 on DEVICE
│   ├── impls axis_torch = a random unit direction, as float32 on DEVICE
│   ├── calls rodrigues_to_matrix(axis_torch, angle_torch)
│   ├── impls R = the rotation it built
│   ├── impls t_torch = a random translation up to ten units, as float32 on DEVICE
│   └── return  # the [4, 4] identity carrying R and t_torch, on DEVICE
└── def create_point_cloud(points: torch.Tensor) -> PointCloud
    ├── # Wraps a coordinate block as a cloud carrying a feature field, so the cases can check what the transform does to fields other than xyz.
    └── calls PointCloud(xyz=points, data={'feat': a ones column matching points in dtype and device})
```
