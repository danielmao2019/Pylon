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
│   ├── impls devices = a list holding the cpu device
│   ├── if cuda is available
│   │   └── impls devices gains the cuda device
│   └── for device in devices
│       ├── impls torch.manual_seed(0)
│       ├── impls points = a [512, 3] float32 standard-normal world-space tensor on device
│       ├── impls generators = a [4, 3, 3] float32 standard-normal tensor on device
│       ├── impls extrinsics = a [4, 4, 4] float32 stack of identities on device
│       ├── impls extrinsics[:, :3, :3] = torch.linalg.matrix_exp(generators - generators.transpose(-1, -2))  # the exponential of a skew-symmetric matrix is a rotation, so the stack holds four distinct camera-to-world poses
│       ├── impls extrinsics[:, :3, 3] = a [4, 3] float32 standard-normal translation on device
│       ├── calls world_to_camera_transform(points=points, extrinsics=extrinsics)  # -> points_camera
│       ├── assert points_camera.shape == (4, 512, 3)
│       └── for index in range(4)
│           ├── calls world_to_camera_transform(points=points, extrinsics=extrinsics[index])  # -> one_pose_points_camera, that one [4, 4] pose mapped on its own
│           ├── if device.type == 'cpu'
│           │   └── assert torch.equal(points_camera[index], one_pose_points_camera)
│           └── else
│               └── assert torch.allclose(points_camera[index], one_pose_points_camera, rtol=1.3e-6, atol=1e-5)  # CUDA inverts and multiplies a stack of several poses with batched kernels that round unlike a single pose's, and the tolerances are torch.testing's float32 defaults
├── def test_world_to_camera_transform_batch_of_one_keeps_its_axis() -> None
│   ├── # A stack of one maps to [1, N, 3] rather than [N, 3], so a caller reading the leading axis is not surprised by a batch of one.
│   ├── impls torch.manual_seed(1)
│   ├── impls points = a [512, 3] float32 standard-normal world-space tensor
│   ├── impls generators = a [1, 3, 3] float32 standard-normal tensor
│   ├── impls extrinsics = a [1, 4, 4] float32 stack of identities
│   ├── impls extrinsics[:, :3, :3] = torch.linalg.matrix_exp(generators - generators.transpose(-1, -2))
│   ├── impls extrinsics[:, :3, 3] = a [1, 3] float32 standard-normal translation
│   ├── calls world_to_camera_transform(points=points, extrinsics=extrinsics)  # -> stacked_points_camera
│   ├── assert stacked_points_camera.shape == (1, 512, 3)
│   ├── calls world_to_camera_transform(points=points, extrinsics=extrinsics[0])  # -> one_pose_points_camera, that same pose as a [4, 4]
│   └── assert one_pose_points_camera.shape == (512, 3) and torch.equal(stacked_points_camera[0], one_pose_points_camera)
├── def test_world_to_camera_transform_inplace_rejects_a_camera_batch() -> None
│   ├── # An unbatched call may write back into its points, but a stack cannot, since [N, 3] in and [B, N, 3] out has no buffer to write into.
│   ├── impls points = a [N, 3] float32 world-space tensor
│   ├── calls world_to_camera_transform(points=points, extrinsics=a [4, 4] float32 pose, inplace=True)
│   ├── assert it returned the same tensor object, its values transformed
│   └── with pytest.raises(AssertionError)
│       └── calls world_to_camera_transform(points=points, extrinsics=a [B, 4, 4] float32 stack, inplace=True)
└── def test_world_to_camera_transform_chunking_matches_the_unchunked_result() -> None
    ├── # Splitting the points into chunks is a memory strategy, not a different computation, so a fixed split returns what one pass returns.
    ├── impls torch.manual_seed(3)
    ├── impls points = a [4096, 3] float32 standard-normal world-space tensor  # large enough to split several ways
    ├── impls generators = a [4, 3, 3] float32 standard-normal tensor
    ├── impls extrinsics = a [4, 4, 4] float32 stack of identities
    ├── impls extrinsics[:, :3, :3] = torch.linalg.matrix_exp(generators - generators.transpose(-1, -2))
    ├── impls extrinsics[:, :3, 3] = a [4, 3] float32 standard-normal translation
    ├── calls world_to_camera_transform(points=points, extrinsics=extrinsics)  # -> unchunked_points_camera
    └── for num_divide in (1, 2, 3)
        ├── calls world_to_camera_transform(points=points, extrinsics=extrinsics, num_divide=num_divide)  # -> chunked_points_camera
        └── assert torch.equal(chunked_points_camera, unchunked_points_camera)
```
