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
│   ├── # A transform carrying a leading batch axis broadcasts over it, so [..., 4, 4] against [N, 3] yields [..., N, 3] and a single [4, 4] still yields [N, 3].
│   ├── # Applies a 4x4 transform to points in homogeneous coordinates, preserving the input's type and batch shape, writing the result back into points when inplace.
│   ├── calls _normalize_points
│   ├── calls _normalize_transform
│   ├── assert not inplace or transform_normalized.ndim == 2  # a transform carrying leading axes yields one copy of the points per axis entry, so there is no single buffer to write back into
│   ├── if isinstance(points_normalized, np.ndarray)
│   │   ├── impls append a ones column, np.matmul by the transform's trailing-two-axes swap, drop the homogeneous coordinate
│   │   ├── if points_was_batched
│   │   │   └── impls add back the batch dimension
│   │   ├── if inplace
│   │   │   ├── impls copy the transformed points into the original points
│   │   │   └── return  # the original numpy points
│   │   └── return  # the transformed numpy points
│   └── else
│       ├── impls points_h = the points with a ones homogeneous column appended
│       ├── calls chunked_matmul(points_h, transform_normalized.transpose(-2, -1), max_divide=max_divide, num_divide=num_divide)  # chunked over the point rows, the transform's leading axes riding through as the small operand's own
│       ├── impls drop the homogeneous coordinate from the chunked-matmul result
│       ├── if points_was_batched
│       │   └── impls add back the batch dimension
│       ├── if inplace
│       │   ├── impls copy the transformed points into the original points
│       │   └── return  # the original torch points
│       └── return  # the transformed torch points
├── def _normalize_points(points: Union[np.ndarray, torch.Tensor]) -> Tuple[Union[np.ndarray, torch.Tensor], bool]
│   ├── # Normalizes points to unbatched (N, 3) while preserving type, reporting whether the input was batched.
│   ├── if points.ndim == 2
│   │   ├── impls normalized_points = points
│   │   ├── impls was_batched = False
│   │   └── return normalized_points, was_batched
│   ├── elif points.ndim == 3
│   │   ├── impls normalized_points = points squeezed on batch axis
│   │   ├── impls was_batched = True
│   │   └── return normalized_points, was_batched
│   └── else
│       └── raise ValueError
├── def _normalize_transform(transform: Union[list, np.ndarray, torch.Tensor], target_type: type, target_dtype: Union[torch.dtype, np.dtype], target_device: Optional[Union[str, torch.device]]) -> Union[np.ndarray, torch.Tensor]
│   ├── # Normalizes a transform to the target type/dtype/device, leaving its leading axes as they came in so a batch of one keeps the axis that names it a batch.
│   ├── if target_type == np.ndarray
│   │   └── calls _normalize_transform_numpy
│   ├── elif target_type == torch.Tensor
│   │   └── calls _normalize_transform_torch
│   ├── else
│   │   └── raise ValueError
│   └── return  # the [..., 4, 4] transform, its leading axes those of the transform it was given
├── def _normalize_transform_numpy(transform: Union[list, np.ndarray, torch.Tensor], target_dtype: np.dtype) -> np.ndarray
│   ├── # Converts a list or tensor transform into a numpy array of the target dtype.
│   ├── if transform is a list
│   │   └── impls transform = an array of transform, of target_dtype
│   ├── if transform is a torch.Tensor
│   │   └── impls transform = that tensor on the cpu, as an array
│   ├── impls transform = transform cast to target_dtype
│   └── return transform  # the numpy transform
└── def _normalize_transform_torch(transform: Union[list, np.ndarray, torch.Tensor], target_dtype: torch.dtype, target_device: torch.device) -> torch.Tensor
    ├── # Converts a list or ndarray transform into a torch tensor on the target dtype and device.
    ├── if transform is a list
    │   └── impls transform = a tensor of transform, of target_dtype on target_device
    ├── if transform is an np.ndarray
    │   └── impls transform = that array as a tensor
    ├── impls transform = transform cast to target_dtype on target_device
    └── return transform  # the torch transform
```

`models/three_d/point_cloud/ops/world_to_camera_transform.py`

```text
world_to_camera_transform.py
├── from typing import Optional
├── import torch
├── from models.three_d.point_cloud.ops.apply_transform import apply_transform
└── def world_to_camera_transform(points: torch.Tensor, extrinsics: torch.Tensor, inplace: bool = False, max_divide: int = 0, num_divide: Optional[int] = None) -> torch.Tensor
    ├── # High-level API mapping world-frame points into the camera frame: inverts the camera-to-world extrinsics and applies them via apply_transform, any leading axes on the extrinsics flowing through onto the result.
    ├── impls world_to_camera = the inverse of the [..., 4, 4] camera-to-world extrinsics, inverted over the trailing two axes
    ├── calls apply_transform(points=points, transform=world_to_camera, inplace=inplace, max_divide=max_divide, num_divide=num_divide)
    └── return  # the [..., N, 3] camera-frame points (the same tensor when inplace); [4, 4] in gives [N, 3] out, [B, 4, 4] gives [B, N, 3]
```
