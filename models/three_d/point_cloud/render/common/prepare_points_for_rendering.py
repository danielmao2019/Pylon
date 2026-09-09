import math
from typing import Callable, Optional, Tuple, Union

import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from models.three_d.point_cloud.ops.world_to_camera_transform import (
    world_to_camera_transform,
)


def _frustum_cull(
    current_points: torch.Tensor,
    bounds_mask: torch.Tensor,
    render_height: int,
    render_width: int,
) -> None:
    """Write into bounds_mask whether each projected point lies within the image bounds.

    Args:
        current_points: [..., N, 3] float torch.Tensor of projected points as
            (x, y, depth), x measured against the width and y against the height.
        bounds_mask: [..., N] bool torch.Tensor written in place with the
            0 <= x < render_width and 0 <= y < render_height test.
        render_height: Target image height in pixels.
        render_width: Target image width in pixels.

    Returns:
        None.
    """
    torch.ge(current_points[..., 0], 0, out=bounds_mask)
    torch.bitwise_and(
        bounds_mask, torch.lt(current_points[..., 0], render_width), out=bounds_mask
    )
    torch.bitwise_and(bounds_mask, torch.ge(current_points[..., 1], 0), out=bounds_mask)
    torch.bitwise_and(
        bounds_mask, torch.lt(current_points[..., 1], render_height), out=bounds_mask
    )


def _prepare_points_for_rendering(
    points: torch.Tensor,
    render_intrinsics: CameraIntrinsics,
    extrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    cull_func: Callable[
        [torch.Tensor, torch.Tensor, int, int],
        None,
    ] = _frustum_cull,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess one chunk of world-space points for rasterization.

    Runs the world-to-camera transform, the positive-depth filter, the
    camera-to-image projection, and the image-bounds cull, marking each survivor
    rather than compacting it out: cameras cull different points, so compaction
    would leave each camera a different length. Point-size expansion is
    intentionally deferred to the renderer, and input validation is handled
    upstream.

    Args:
        points: [N, 3] float torch.Tensor of world-space coordinates.
        render_intrinsics: CameraIntrinsics carrying the camera-to-image
            projection; its params broadcast against the point axis.
        extrinsics: [..., 4, 4] float torch.Tensor of camera-to-world matrices in
            the OpenCV convention, one per camera carried.
        resolution: Target image resolution as an (H, W) tuple.
        cull_func: Callable writing the image-bounds test of its projected points
            into its bounds_mask in place.

    Returns:
        A (points_2d, valid) tuple where points_2d is a [..., N, 3] float
        torch.Tensor of (x, y, depth) per camera and valid is the [..., N] bool
        torch.Tensor marking the points each camera keeps.
    """
    # Resolution is consistently (H, W). x uses W, y uses H.
    render_height, render_width = resolution

    # Transform world-space -> camera coordinates (OpenCV convention), the
    # extrinsics' leading axes flowing through onto the result.
    current_points = world_to_camera_transform(points=points, extrinsics=extrinsics)

    # Mark the points with positive depth (in front of camera).
    valid = current_points[..., 2] > 0

    # Project to pixel coordinates in place. Output: [x, y, depth]
    render_intrinsics.project(points_camera=current_points, inplace=True)

    # Mark the points inside the image bounds.
    bounds_mask = torch.empty(
        current_points.shape[:-1], dtype=torch.bool, device=current_points.device
    )
    cull_func(
        current_points=current_points,
        bounds_mask=bounds_mask,
        render_height=render_height,
        render_width=render_width,
    )

    valid = valid & bounds_mask
    return current_points, valid


def _prepare_points_for_rendering_chunked(
    points: torch.Tensor,
    camera: Union[Camera, Cameras],
    resolution: Tuple[int, int],
    chunk_size: int = 2048,
    cull_func: Callable[
        [torch.Tensor, torch.Tensor, int, int],
        None,
    ] = _frustum_cull,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run _prepare_points_for_rendering over fixed-size point chunks and concatenate them.

    The chunking is over the point axis only, so a camera batch is never split,
    and the concatenation leaves the point axis in its input order so row i still
    names point i.

    Args:
        points: [N, 3] float torch.Tensor of world-space coordinates.
        camera: The Camera or Cameras to render through, already brought to the
            OpenCV pose frame and scaled to resolution.
        resolution: Target image resolution as an (H, W) tuple.
        chunk_size: Number of points preprocessed per chunk.
        cull_func: Callable writing the image-bounds test of its projected points
            into its bounds_mask in place.

    Returns:
        A (points_2d, valid) tuple where points_2d is a [..., N, 3] float
        torch.Tensor of (x, y, depth) with the point axis in the order of points
        and valid is the [..., N] bool torch.Tensor marking the points each camera
        keeps, in that same order.

    Raises:
        AssertionError: If no point survived culling for any camera.
    """
    render_intrinsics = camera.intrinsics
    extrinsics = camera.extrinsics.extrinsics
    N = points.shape[0]

    points_chunks = []
    valid_chunks = []
    for i in range(0, N, chunk_size):
        j = min(N, i + chunk_size)
        chunk_points, chunk_valid = _prepare_points_for_rendering(
            points=points[i:j],
            render_intrinsics=render_intrinsics,
            extrinsics=extrinsics,
            resolution=resolution,
            cull_func=cull_func,
        )
        points_chunks.append(chunk_points)
        valid_chunks.append(chunk_valid)

    if not any(bool(chunk_valid.any()) for chunk_valid in valid_chunks):
        raise AssertionError(
            "No points remained after culling in all chunks. "
            f"{N=} {resolution=} {extrinsics.shape=} {len(valid_chunks)=}"
        )

    return torch.cat(points_chunks, dim=-2), torch.cat(valid_chunks, dim=-1)


def prepare_points_for_rendering(
    pc: PointCloud,
    camera: Union[Camera, Cameras],
    resolution: Tuple[int, int],
    max_divide: int = 0,
    num_divide: Optional[int] = None,
    cull_func: Callable[
        [torch.Tensor, torch.Tensor, int, int],
        None,
    ] = _frustum_cull,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare a point cloud for rasterization through one camera or a batch of them.

    Brings the camera to the OpenCV pose frame and the target resolution, then
    adaptively chunks the point preprocessing to mitigate CUDA OOM. Row i of the
    returned points is point i of pc.xyz, so a per-point attribute is looked up by
    the same index a rasterizer resolves per pixel.

    Args:
        pc: PointCloud whose xyz carries the [N, 3] world-space points.
        camera: The Camera (no leading axis) or Cameras (a [B] leading axis) to
            render through.
        resolution: Target image resolution as an (H, W) tuple.
        max_divide: Maximum number of times the point chunk may be halved on CUDA
            OOM before the error is re-raised.
        num_divide: If not None, the fixed number of chunk halvings, with no OOM
            retry.
        cull_func: Callable writing the image-bounds test of its projected points
            into its bounds_mask in place.

    Returns:
        A (points_2d, valid) tuple where points_2d is a [..., N, 3] float
        torch.Tensor of (x, y, depth) with the point axis in pc.xyz order and
        valid is the [..., N] bool torch.Tensor marking the points each camera
        keeps; a Camera gives [N, 3] / [N] and a Cameras gives [B, N, 3] / [B, N].

    Raises:
        torch.cuda.OutOfMemoryError: If the chunk is still too large after
            max_divide halvings.
    """
    assert isinstance(pc, PointCloud), f"{type(pc)=}"
    assert isinstance(camera, (Camera, Cameras)), f"{type(camera)=}"
    points = pc.xyz

    camera_prepared = camera.to(
        device=points.device, extr_convention="opencv"
    ).scale_intrinsics(resolution=resolution)

    # If `num_divide` is set, derive chunk size from N / 2**num_divide.
    N = points.shape[0]
    if num_divide is not None:
        chunk_size = max(1, math.ceil(N / (2**num_divide)))
        return _prepare_points_for_rendering_chunked(
            points=points,
            camera=camera_prepared,
            resolution=resolution,
            chunk_size=chunk_size,
            cull_func=cull_func,
        )

    # Otherwise, progressively halve chunk size on CUDA OOM up to `max_divide`.
    n = 0
    while n <= max_divide:
        chunk_size = max(1, math.ceil(N / (2**n)))
        try:
            return _prepare_points_for_rendering_chunked(
                points=points,
                camera=camera_prepared,
                resolution=resolution,
                chunk_size=chunk_size,
                cull_func=cull_func,
            )
        except torch.cuda.OutOfMemoryError:
            n += 1
            torch.cuda.empty_cache()
            continue

    raise torch.cuda.OutOfMemoryError(
        f"CUDA OOM after {max_divide} divisions in prepare_points_for_rendering."
    )
