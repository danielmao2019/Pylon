# `models/three_d/point_cloud/ops/` code skeleton

## Code implementation structure

`models/three_d/point_cloud/ops/apply_transform.py`

```text
apply_transform.py
├── import numpy as np
├── import torch
├── from typing import Optional, Tuple, Union
├── from utils.ops.chunked_matmul import chunked_matmul
├── def apply_transform(points: Union[np.ndarray, torch.Tensor], transform: Union[list, np.ndarray, torch.Tensor], inplace: bool = False, max_divide: int = 0, num_divide: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]
│   ├── # Applies a 4x4 transform to points in homogeneous coordinates, preserving the input's type and batch shape, writing the result back into points when inplace.
│   ├── calls _normalize_points
│   ├── calls _normalize_transform
│   ├── if isinstance(points_normalized, np.ndarray)
│   │   ├── impls append a ones column, np.dot by the transform transpose, drop the homogeneous coordinate
│   │   ├── if points_was_batched
│   │   │   └── impls add back the batch dimension
│   │   ├── if inplace
│   │   │   ├── impls copy the transformed points into the original points
│   │   │   └── return  # the original numpy points
│   │   └── return  # the transformed numpy points
│   └── else
│       ├── impls append a ones homogeneous column to the points
│       ├── calls chunked_matmul  # homogeneous points by the transform transpose, passing max_divide and num_divide; then drop the homogeneous coordinate
│       ├── if points_was_batched
│       │   └── impls add back the batch dimension
│       ├── if inplace
│       │   ├── impls copy the transformed points into the original points
│       │   └── return  # the original torch points
│       └── return  # the transformed torch points
├── def _normalize_points(points: Union[np.ndarray, torch.Tensor]) -> Tuple[Union[np.ndarray, torch.Tensor], bool]
│   ├── # Normalizes points to unbatched (N, 3) while preserving type, reporting whether the input was batched.
│   ├── if points.ndim == 2
│   │   └── return  # (points, False)
│   ├── elif points.ndim == 3
│   │   └── return  # (points.squeeze(0), True)
│   └── else
│       └── raise ValueError
├── def _normalize_transform(transform: Union[list, np.ndarray, torch.Tensor], target_type: type, target_dtype: Union[torch.dtype, np.dtype], target_device: Optional[Union[str, torch.device]]) -> Union[np.ndarray, torch.Tensor]
│   ├── # Normalizes a transform to the target type/dtype/device and squeezes it to a [4, 4] matrix.
│   ├── if target_type == np.ndarray
│   │   └── calls _normalize_transform_numpy
│   ├── elif target_type == torch.Tensor
│   │   └── calls _normalize_transform_torch
│   ├── else
│   │   └── raise ValueError
│   └── return  # the squeezed [4, 4] transform
├── def _normalize_transform_numpy(transform: Union[list, np.ndarray, torch.Tensor], target_dtype: np.dtype) -> np.ndarray
│   ├── # Converts a list or tensor transform into a numpy array of the target dtype.
│   └── return  # the numpy transform
└── def _normalize_transform_torch(transform: Union[list, np.ndarray, torch.Tensor], target_dtype: torch.dtype, target_device: torch.device) -> torch.Tensor
    ├── # Converts a list or ndarray transform into a torch tensor on the target dtype and device.
    └── return  # the torch transform
```

`models/three_d/point_cloud/ops/world_to_camera_transform.py`

```text
world_to_camera_transform.py
├── from typing import Optional
├── import torch
├── from models.three_d.point_cloud.ops.apply_transform import apply_transform
└── def world_to_camera_transform(points: torch.Tensor, extrinsics: torch.Tensor, inplace: bool = False, max_divide: int = 0, num_divide: Optional[int] = None) -> torch.Tensor
    ├── # High-level API mapping world-frame points into the camera frame: builds the world-to-camera 4x4 matrix from the inverse camera-to-world extrinsic and applies it via apply_transform.
    ├── impls invert the camera-to-world extrinsics into the world-to-camera 4x4 matrix
    ├── calls apply_transform  # the points by the world-to-camera matrix, forwarding inplace, max_divide, and num_divide
    └── return  # the [N, 3] camera-frame points (the same tensor when inplace)
```
