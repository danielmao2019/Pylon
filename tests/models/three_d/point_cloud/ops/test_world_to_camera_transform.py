import pytest
import torch

from models.three_d.point_cloud.ops.world_to_camera_transform import (
    world_to_camera_transform,
)


def test_world_to_camera_transform_carries_the_camera_batch_axis() -> None:
    """A stack of extrinsics maps one cloud through every pose in one call, each slice equal to what that pose maps on its own, which is the contract the batched renderer rests on.

    Args:
        None.

    Returns:
        None.
    """
    torch.manual_seed(0)
    points = torch.randn(size=(512, 3), dtype=torch.float32)
    generators = torch.randn(size=(4, 3, 3), dtype=torch.float32)
    extrinsics = torch.eye(4, dtype=torch.float32).repeat(4, 1, 1)
    extrinsics[:, :3, :3] = torch.linalg.matrix_exp(
        generators - generators.transpose(-1, -2)
    )
    extrinsics[:, :3, 3] = torch.randn(size=(4, 3), dtype=torch.float32)

    points_camera = world_to_camera_transform(points=points, extrinsics=extrinsics)

    assert points_camera.shape == (4, 512, 3), (
        "Expected a four-pose stack to map the cloud into a [B, N, 3] result. "
        f"{points_camera.shape=} {points.shape=} {extrinsics.shape=}"
    )

    for index in range(4):
        one_pose_points_camera = world_to_camera_transform(
            points=points, extrinsics=extrinsics[index]
        )
        assert torch.equal(points_camera[index], one_pose_points_camera), (
            "Expected the batched result's slice to equal what that pose maps on its "
            f"own. {index=} {points_camera[index]=} {one_pose_points_camera=}"
        )


def test_world_to_camera_transform_batch_of_one_keeps_its_axis() -> None:
    """A stack of one maps to [1, N, 3] rather than [N, 3], so a caller reading the leading axis is not surprised by a batch of one.

    Args:
        None.

    Returns:
        None.
    """
    torch.manual_seed(1)
    points = torch.randn(size=(512, 3), dtype=torch.float32)
    generators = torch.randn(size=(1, 3, 3), dtype=torch.float32)
    extrinsics = torch.eye(4, dtype=torch.float32).repeat(1, 1, 1)
    extrinsics[:, :3, :3] = torch.linalg.matrix_exp(
        generators - generators.transpose(-1, -2)
    )
    extrinsics[:, :3, 3] = torch.randn(size=(1, 3), dtype=torch.float32)

    stacked_points_camera = world_to_camera_transform(
        points=points, extrinsics=extrinsics
    )

    assert stacked_points_camera.shape == (1, 512, 3), (
        "Expected a stack of one to keep its leading axis. "
        f"{stacked_points_camera.shape=} {points.shape=} {extrinsics.shape=}"
    )

    one_pose_points_camera = world_to_camera_transform(
        points=points, extrinsics=extrinsics[0]
    )

    assert one_pose_points_camera.shape == (512, 3) and torch.equal(
        stacked_points_camera[0], one_pose_points_camera
    ), (
        "Expected the same pose stated as a [4, 4] to map to [N, 3] equal to the "
        f"stack's only slice. {one_pose_points_camera.shape=} "
        f"{stacked_points_camera[0]=} {one_pose_points_camera=}"
    )


def test_world_to_camera_transform_inplace_rejects_a_camera_batch() -> None:
    """An unbatched call may write back into its points, but a stack cannot, since [N, 3] in and [B, N, 3] out has no buffer to write into.

    Args:
        None.

    Returns:
        None.
    """
    torch.manual_seed(2)
    points = torch.randn(size=(512, 3), dtype=torch.float32)
    points_world = points.clone()
    generators = torch.randn(size=(4, 3, 3), dtype=torch.float32)
    extrinsics = torch.eye(4, dtype=torch.float32).repeat(4, 1, 1)
    extrinsics[:, :3, :3] = torch.linalg.matrix_exp(
        generators - generators.transpose(-1, -2)
    )
    extrinsics[:, :3, 3] = torch.randn(size=(4, 3), dtype=torch.float32)

    points_camera = world_to_camera_transform(
        points=points, extrinsics=extrinsics[0], inplace=True
    )

    assert points_camera is points and not torch.equal(points_camera, points_world), (
        "Expected an unbatched inplace call to return the points object itself with "
        f"its values transformed. {points_camera is points=} "
        f"{points_camera=} {points_world=}"
    )

    with pytest.raises(AssertionError):
        world_to_camera_transform(points=points, extrinsics=extrinsics, inplace=True)


def test_world_to_camera_transform_chunking_matches_the_unchunked_result() -> None:
    """Splitting the points into chunks is a memory strategy, not a different computation, so a fixed split returns what one pass returns.

    Args:
        None.

    Returns:
        None.
    """
    torch.manual_seed(3)
    points = torch.randn(size=(4096, 3), dtype=torch.float32)
    generators = torch.randn(size=(4, 3, 3), dtype=torch.float32)
    extrinsics = torch.eye(4, dtype=torch.float32).repeat(4, 1, 1)
    extrinsics[:, :3, :3] = torch.linalg.matrix_exp(
        generators - generators.transpose(-1, -2)
    )
    extrinsics[:, :3, 3] = torch.randn(size=(4, 3), dtype=torch.float32)

    unchunked_points_camera = world_to_camera_transform(
        points=points, extrinsics=extrinsics
    )

    for num_divide in (1, 2, 3):
        chunked_points_camera = world_to_camera_transform(
            points=points, extrinsics=extrinsics, num_divide=num_divide
        )
        assert torch.equal(chunked_points_camera, unchunked_points_camera), (
            "Expected a fixed chunk split to return the unchunked result elementwise. "
            f"{num_divide=} {chunked_points_camera=} {unchunked_points_camera=}"
        )
