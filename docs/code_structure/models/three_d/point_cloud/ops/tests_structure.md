# `models/three_d/point_cloud/ops/` tests skeleton

## Tests implementation structure

`tests/models/three_d/point_cloud/ops/test_world_to_camera_transform.py`

```text
test_world_to_camera_transform.py
├── import pytest
├── import torch
├── from models.three_d.point_cloud.ops.world_to_camera_transform import world_to_camera_transform
├── def test_world_to_camera_transform_carries_the_camera_batch_axis() -> None
│   ├── # A stack of extrinsics maps one cloud through every pose in one call, each slice equal to what that pose maps on its own, bit for bit on cpu and within floating-point rounding on cuda, which is the contract the batched renderer rests on.
│   ├── for each device of cpu, and cuda when it is available
│   │   ├── impls points = a [N, 3] float32 world-space tensor on that device
│   │   ├── impls extrinsics = a [B, 4, 4] float32 stack of distinct camera-to-world poses on that device
│   │   ├── calls world_to_camera_transform(points=points, extrinsics=extrinsics)
│   │   ├── assert the result is [B, N, 3]
│   │   └── for each pose the stack carries
│   │       ├── calls world_to_camera_transform(points=points, extrinsics=that one [4, 4] pose)
│   │       ├── if device is cpu
│   │       │   └── assert the batched result's matching slice equals it exactly
│   │       └── else
│   │           └── assert the batched result's matching slice agrees with it within floating-point rounding  # CUDA inverts and multiplies a stack of several poses with batched kernels that round unlike a single pose's
│   └── return
├── def test_world_to_camera_transform_batch_of_one_keeps_its_axis() -> None
│   ├── # A stack of one maps to [1, N, 3] rather than [N, 3], so a caller reading the leading axis is not surprised by a batch of one.
│   ├── impls points = a [N, 3] float32 world-space tensor
│   ├── calls world_to_camera_transform(points=points, extrinsics=a [1, 4, 4] float32 stack)
│   ├── assert the result is [1, N, 3]
│   ├── calls world_to_camera_transform(points=points, extrinsics=that same pose as a [4, 4])
│   ├── assert that result is [N, 3] and equals the stacked result's only slice
│   └── return
├── def test_world_to_camera_transform_inplace_rejects_a_camera_batch() -> None
│   ├── # An unbatched call may write back into its points, but a stack cannot, since [N, 3] in and [B, N, 3] out has no buffer to write into.
│   ├── impls points = a [N, 3] float32 world-space tensor
│   ├── calls world_to_camera_transform(points=points, extrinsics=a [4, 4] float32 pose, inplace=True)
│   ├── assert it returned the same tensor object, its values transformed
│   └── with pytest.raises(AssertionError)
│       └── calls world_to_camera_transform(points=points, extrinsics=a [B, 4, 4] float32 stack, inplace=True)
└── def test_world_to_camera_transform_chunking_matches_the_unchunked_result() -> None
    ├── # Splitting the points into chunks is a memory strategy, not a different computation, so a fixed split returns what one pass returns.
    ├── impls points = a [N, 3] float32 world-space tensor large enough to split several ways
    ├── calls world_to_camera_transform(points=points, extrinsics=a [B, 4, 4] float32 stack)
    ├── for each of several num_divide settings
    │   ├── calls world_to_camera_transform(points=points, extrinsics=that same stack, num_divide=that setting)
    │   └── assert it equals the unchunked result elementwise
    └── return
```
