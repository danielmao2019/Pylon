# data/structures/three_d/mesh/ops — code structure

`data/structures/three_d/mesh/ops/apply_transform.py`

```text
apply_transform.py
├── from typing import Optional
├── import torch
├── from data.structures.three_d.mesh.mesh import Mesh
├── from utils.ops.chunked_matmul import chunked_matmul
└── def apply_transform(mesh: Mesh, transform: torch.Tensor, max_divide: int = 0, num_divide: Optional[int] = None) -> Mesh
    ├── # Returns a copy of mesh whose verts are mapped through a 4x4 transform in homogeneous coordinates, leaving faces and texture unchanged.
    ├── impls build the homogeneous [V, 4] verts by appending a ones column to mesh.verts
    ├── calls chunked_matmul  # homogeneous verts by transform.T, passing max_divide and num_divide, chunked over the V rows
    ├── impls drop the homogeneous coordinate to get the [V, 3] transformed verts
    ├── calls Mesh  # rebuild with the transformed verts, original faces, original texture
    └── return  # the transformed Mesh
```

`data/structures/three_d/mesh/ops/world_to_camera_transform.py`

```text
world_to_camera_transform.py
├── from typing import Optional
├── import torch
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.ops.apply_transform import apply_transform
└── def world_to_camera_transform(mesh: Mesh, extrinsics: torch.Tensor, max_divide: int = 0, num_divide: Optional[int] = None) -> Mesh
    ├── # High-level API mapping a mesh's verts from world into the camera frame: builds the world-to-camera 4x4 matrix from the inverse camera-to-world extrinsic and applies it via apply_transform.
    ├── impls invert the camera-to-world extrinsics into the world-to-camera 4x4 matrix
    ├── calls apply_transform  # the mesh by the world-to-camera matrix, passing max_divide and num_divide
    └── return  # the camera-frame Mesh
```
