# Vision-3D Transforms Code Structure

## Code structure trees

`data/transforms/vision_3d/random_rigid_transform.py`

```text
random_rigid_transform.py
├── from typing import Any, Optional, Tuple
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.extrinsics.rotation.rodrigues import rodrigues_to_matrix
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.transforms.base_transform import BaseTransform
├── from models.three_d.point_cloud.ops import apply_transform
└── class RandomRigidTransform(BaseTransform)
    ├── # Poses the source cloud of a registration pair by a sampled rigid transform, so the ground truth the pair carries is exercised rather than memorized.
    ├── def __call__(self, src_pc: PointCloud, tgt_pc: PointCloud, transform: torch.Tensor, seed: Optional[Any] = None) -> Tuple[PointCloud, PointCloud, torch.Tensor]
    │   ├── # Poses the source cloud and hands back the pair with the ground truth adjusted for that pose.
    │   ├── calls self._get_generator(g_type='torch', seed=seed)
    │   ├── impls generator = the generator it built  # the seed is the caller's, so one datapoint poses identically across epochs
    │   ├── calls self._sample_rigid_transform(transform.device, generator)
    │   ├── impls random_transform = the transform it sampled
    │   ├── calls apply_transform(points=src_pc.xyz, transform=random_transform)
    │   ├── impls transformed_src_xyz = the posed coordinates
    │   ├── impls src_fields = transformed_src_xyz under 'xyz' followed by every other field of src_pc in its own order
    │   ├── calls PointCloud(data=src_fields, meta_data=src_pc.meta_data)
    │   ├── impls new_src_pc = the cloud it built  # a pose is not a source, so the meta data crosses unchanged rather than being rebuilt from the posed tensors
    │   ├── impls tgt_fields = tgt_pc.xyz under 'xyz' followed by every other field of tgt_pc in its own order
    │   ├── calls PointCloud(data=tgt_fields, meta_data=tgt_pc.meta_data)
    │   ├── impls new_tgt_pc = the cloud it built  # the target is rebuilt rather than passed through, so neither returned cloud aliases an input
    │   ├── impls random_transform_inv = the inverse of random_transform
    │   ├── impls new_transform = transform composed with random_transform_inv, mapping the posed source onto the target
    │   └── return new_src_pc, new_tgt_pc, new_transform
    ├── def _sample_rigid_transform(self, device: torch.device, generator: torch.Generator) -> torch.Tensor
    │   ├── # Samples the one [4, 4] rigid transform a call poses the source by.
    │   ├── if self.method is Rodrigues
    │   │   └── calls self._sample_rotation_Rodrigues(device, generator)
    │   ├── else
    │   │   └── calls self._sample_rotation_Euler(device, generator)
    │   ├── impls R = the rotation it sampled
    │   ├── impls trans_dir = a random three-vector on device from generator, normalized to unit length
    │   ├── impls trans_mag = a magnitude drawn uniformly from zero to self.trans_mag
    │   ├── impls trans = trans_dir scaled by trans_mag
    │   ├── impls transform = the [4, 4] identity on device, with R in its rotation block and trans in its translation column  # impls-node-one-step:skip — one step; the "and" names what it is made of
    │   └── return transform
    ├── def _sample_rotation_Rodrigues(self, device: torch.device, generator: torch.Generator) -> torch.Tensor
    │   ├── # Samples one [3, 3] rotation about a single random axis.
    │   ├── impls rot_mag_rad = self.rot_mag in radians, by np.radians
    │   ├── impls axis = a random three-vector on device from generator, normalized to unit length
    │   ├── impls angle = a scalar drawn uniformly from minus rot_mag_rad to rot_mag_rad
    │   ├── calls rodrigues_to_matrix(axis, angle)
    │   └── return  # the rotation matrix it built
    └── def _sample_rotation_Euler(self, device: torch.device, generator: torch.Generator) -> torch.Tensor
        ├── # Samples one [3, 3] rotation from per-axis Euler angles, over as many axes as self.num_axis names.
        ├── if self.num_axis is zero
        │   └── return  # the [3, 3] identity on device
        ├── impls rot_mag_rad = self.rot_mag in radians, by np.radians
        ├── impls angles = three angles drawn uniformly from minus rot_mag_rad to rot_mag_rad
        ├── impls Rx, Ry, Rz = the three axis rotations those angles make, on device
        ├── if self.num_axis is one
        │   └── return Rz  # return-nodes-no-construction:skip — Rz is the local single-axis rotation this branch already built, not a return type
        └── return  # Rx composed with Ry composed with Rz
```
