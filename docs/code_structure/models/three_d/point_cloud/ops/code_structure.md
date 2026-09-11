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
│   ├── def _validate_inputs [local]
│   │   ├── assert points is an np.ndarray or a torch.Tensor
│   │   ├── assert points is [N, 3], or [1, N, 3] carrying one leading batch axis of one
│   │   ├── assert transform is a list, an np.ndarray or a torch.Tensor
│   │   ├── assert transform is [4, 4], or a [..., 4, 4] stack carrying leading axes
│   │   ├── assert inplace is a bool
│   │   └── if inplace
│   │       └── assert transform carries no leading axes  # they yield one copy of the points per entry, leaving no single buffer to write back into
│   ├── calls _validate_inputs()
│   ├── def _normalize_inputs [local]
│   │   ├── calls _normalize_points(points=points)
│   │   ├── impls points, was_batched = the returned values from _normalize_points  # an unbatched [N, 3], a view of the caller's array when a batch axis was squeezed
│   │   ├── calls _normalize_transform(transform=transform, target_type=type(points), target_dtype=points.dtype, target_device=the points' device when a torch.Tensor, else None)
│   │   ├── impls transform = the [..., 4, 4] transform it returned, in the points' own type, dtype and device
│   │   ├── assert transform.dtype == points.dtype  # the output check: the normalized points and transform share one dtype
│   │   └── return points, was_batched, transform
│   ├── calls _normalize_inputs(points=points, transform=transform)
│   ├── impls points, was_batched, transform = the returned values from _normalize_inputs
│   ├── if isinstance(points, np.ndarray)
│   │   ├── impls transformed = points with a ones column appended, np.matmul'd by transform transposed over its trailing two axes, the homogeneous coordinate dropped  # impls-node-one-step:skip; broadcasts over the transform's leading axes: [..., 4, 4] yields [..., N, 3], [4, 4] still [N, 3]
│   │   ├── if inplace
│   │   │   ├── impls copy transformed into points  # points is the caller's array, or a view of it when a batch axis was squeezed, so the write lands in the caller's own buffer
│   │   │   └── impls transformed = points
│   │   ├── if was_batched
│   │   │   └── impls transformed = transformed with the batch dimension added back
│   │   └── return transformed  # the numpy points, the caller's own array when inplace and unbatched
│   └── else
│       ├── impls points_h = points with a ones homogeneous column appended
│       ├── calls chunked_matmul(large=points_h, small=transform transposed over its trailing two axes, max_divide=max_divide, num_divide=num_divide)  # chunked over the point rows, broadcast over the transform's leading axes: [..., 4, 4] yields [..., N, 4], [4, 4] still [N, 4]
│       ├── impls transformed = the chunked-matmul result with the homogeneous coordinate dropped
│       ├── if inplace
│       │   ├── impls copy transformed into points  # points is the caller's tensor, or a view of it when a batch axis was squeezed, so the write lands in the caller's own buffer
│       │   └── impls transformed = points
│       ├── if was_batched
│       │   └── impls transformed = transformed with the batch dimension added back
│       └── return transformed  # the torch points, the caller's own tensor when inplace and unbatched
├── def _normalize_points(points: Union[np.ndarray, torch.Tensor]) -> Tuple[Union[np.ndarray, torch.Tensor], bool]
│   ├── # Normalizes points to unbatched (N, 3) while preserving type, reporting whether the input was batched.
│   ├── if points.ndim == 2
│   │   ├── impls normalized_points = points
│   │   ├── impls was_batched = False
│   │   └── return normalized_points, was_batched
│   └── else
│       ├── impls normalized_points = points squeezed on batch axis  # a view, so an inplace write through it lands in the caller's array
│       ├── impls was_batched = True
│       └── return normalized_points, was_batched
├── def _normalize_transform(transform: Union[list, np.ndarray, torch.Tensor], target_type: type, target_dtype: Union[torch.dtype, np.dtype], target_device: Optional[Union[str, torch.device]]) -> Union[np.ndarray, torch.Tensor]
│   ├── # Normalizes a transform to the target type/dtype/device, leaving its leading axes as they came in.
│   ├── if target_type == np.ndarray
│   │   └── calls _normalize_transform_numpy
│   ├── elif target_type == torch.Tensor
│   │   └── calls _normalize_transform_torch
│   ├── else
│   │   └── raise ValueError
│   └── return  # the [..., 4, 4] transform, its leading axes as they came in
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
    ├── def _validate_inputs [local]
    │   ├── impls asserts points is a [N, 3] float torch.Tensor  # the point axis is the only one this entry takes; a leading axis on the points would compose with the extrinsics' own and leave the output's axis order unstated
    │   ├── impls asserts extrinsics is a [..., 4, 4] float torch.Tensor on the points' device
    │   └── impls asserts inplace is False whenever extrinsics carries a leading axis  # [N, 3] in and [..., N, 3] out is a shape expansion, so there is no buffer to write back into
    ├── calls _validate_inputs()
    ├── impls world_to_camera = the inverse of the [..., 4, 4] camera-to-world extrinsics, inverted over the trailing two axes  # one op over the whole stack, a single pose and a batch of poses alike
    ├── calls apply_transform(points=points, transform=world_to_camera, inplace=inplace, max_divide=max_divide, num_divide=num_divide)
    └── return  # the [..., N, 3] camera-frame points; [4, 4] in gives [N, 3] out and is the only case inplace returns the same tensor, [B, 4, 4] gives [B, N, 3]
```
