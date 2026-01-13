import math
from typing import Callable, Optional, Tuple

import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.point_cloud.camera.project import project_3d_to_2d
from data.structures.three_d.point_cloud.camera.transform import (
    world_to_camera_transform,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def _frustum_cull(
    current_points: torch.Tensor,
    bounds_mask: torch.Tensor,
    render_height: int,
    render_width: int,
) -> None:
    torch.ge(current_points[:, 0], 0, out=bounds_mask)
    torch.bitwise_and(
        bounds_mask, torch.lt(current_points[:, 0], render_width), out=bounds_mask
    )
    torch.bitwise_and(bounds_mask, torch.ge(current_points[:, 1], 0), out=bounds_mask)
    torch.bitwise_and(
        bounds_mask, torch.lt(current_points[:, 1], render_height), out=bounds_mask
    )


def _prepare_points_for_rendering(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    cull_func: Callable[
        [torch.Tensor, torch.Tensor, int, int],
        None,
    ] = _frustum_cull,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess a single chunk of points for rasterization.

    Performs: world→camera transform, perspective projection, and image-bounds
    filtering. Point-size expansion is intentionally deferred to the renderer.
    Input validation should be handled upstream.

    Args:
        points: [N, 3] world-space coordinates
        intrinsics: [3, 3] camera intrinsics
        extrinsics: [4, 4] camera extrinsics
        resolution: (H, W) target image resolution

    Returns:
        (points_2d, indices) where:
        - points_2d: [M, 3] with (x, y, depth) for in-bounds points
        - indices: [M] original indices into `points`

    Raises:
        AssertionError if nothing survives visibility/bounds tests
    """
    # Resolution is consistently (H, W). x uses W, y uses H.
    render_height, render_width = resolution
    # Track total input size once.
    num_points = points.shape[0]

    # Clone points; allocate index map. Delay large temps until after culling.
    points_work = points.clone()
    indices_work = torch.arange(num_points, device=points.device)

    # Preallocate a reusable boolean mask; slice per active range.
    mask_buffer = torch.empty(num_points, dtype=torch.bool, device=points.device)

    # Intrinsics/extrinsics are already prepared by the public entrypoint.

    # Transform world-space → camera coordinates (OpenCV convention).
    # Operate on the active prefix to keep memory compact.
    valid_count = num_points
    current_points = points_work[:valid_count]

    world_to_camera_transform(
        points=current_points, extrinsics=extrinsics, inplace=True, max_divide=2
    )

    # Retain only points with positive depth (in front of camera).
    depth_mask = mask_buffer[:valid_count]
    torch.gt(current_points[:, 2], 0, out=depth_mask)
    depth_valid = int(depth_mask.sum().item())
    if depth_valid == 0:
        # Allow callers to handle empty batches; global culling logic will surface
        # an error if *all* batches end up empty.
        return current_points[:0], indices_work[:0]

    # Compact surviving points and indices.
    idx_depth = torch.nonzero(depth_mask, as_tuple=False).squeeze(1)
    points_scratch = torch.index_select(current_points, 0, idx_depth)
    indices_scratch = torch.index_select(indices_work[:valid_count], 0, idx_depth)

    valid_count = depth_valid
    points_work = points_scratch
    indices_work = indices_scratch
    current_points = points_work[:valid_count]

    # Project to pixel coordinates in place. Output: [x, y, depth]
    project_3d_to_2d(points=current_points, intrinsics=intrinsics, inplace=True)

    # No point-size expansion here; keep memory footprint minimal.

    # Remove points outside image bounds.
    bounds_mask = mask_buffer[:valid_count]
    cull_func(
        current_points=current_points,
        bounds_mask=bounds_mask,
        render_height=render_height,
        render_width=render_width,
    )

    bounds_valid = int(bounds_mask.sum().item())
    if bounds_valid == 0:
        # Same reasoning as the depth filter: individual batches may legitimately
        # have no in-bounds points even though other batches do.
        return current_points[:0], indices_work[:0]

    # Compact in-bounds points and associated indices.
    idx_bounds = torch.nonzero(bounds_mask, as_tuple=False).squeeze(1)
    points_scratch = torch.index_select(current_points, 0, idx_bounds)
    indices_scratch = torch.index_select(indices_work[:valid_count], 0, idx_bounds)

    valid_count = bounds_valid
    points_work = points_scratch
    indices_work = indices_scratch

    return points_work[:valid_count], indices_work[:valid_count]


def _prepare_points_for_rendering_batched(
    points: torch.Tensor,
    camera: Camera,
    resolution: Tuple[int, int],
    batch_size: int = 2048,
    cull_func: Callable[
        [torch.Tensor, torch.Tensor, int, int],
        None,
    ] = _frustum_cull,
):
    """Batch `_prepare_points_for_rendering`, then merge and depth-sort.

    Returns (points_2d, indices) after a global back-to-front sort by depth.
    """
    intrinsics = camera.intrinsics
    extrinsics = camera.extrinsics
    N = points.shape[0]
    outputs = []
    idx_outputs = []
    for i in range(0, N, batch_size):
        j = min(N, i + batch_size)
        pts, idx = _prepare_points_for_rendering(
            points=points[i:j],
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            resolution=resolution,
            cull_func=cull_func,
        )
        if pts.numel() == 0:
            continue
        outputs.append(pts)
        # Offset per-batch indices by the batch start to produce
        # global indices into the original point array.
        idx_outputs.append(idx + i)

    if not outputs:
        raise AssertionError("No points remained after culling in all batches")

    pts_all = torch.cat(outputs, dim=0)
    idx_all = torch.cat(idx_outputs, dim=0)

    # Drop references to per-batch tensors so the allocator can reclaim memory.
    outputs = None
    idx_outputs = None
    torch.cuda.empty_cache()

    # Global back-to-front depth sort using out parameters.
    valid_count = pts_all.shape[0]
    sort_values = torch.empty(valid_count, dtype=pts_all.dtype, device=pts_all.device)
    sort_indices = torch.empty(valid_count, dtype=torch.long, device=pts_all.device)
    torch.sort(pts_all[:, 2], dim=0, descending=True, out=(sort_values, sort_indices))
    points_sorted = torch.empty_like(pts_all)
    indices_sorted = torch.empty_like(idx_all)
    torch.index_select(pts_all, 0, sort_indices, out=points_sorted)
    torch.index_select(idx_all, 0, sort_indices, out=indices_sorted)
    return points_sorted, indices_sorted


def prepare_points_for_rendering(
    pc: PointCloud,
    camera: Camera,
    resolution: Tuple[int, int],
    max_divide: int = 0,
    num_divide: Optional[int] = None,
    cull_func: Callable[
        [torch.Tensor, torch.Tensor, int, int],
        None,
    ] = _frustum_cull,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Public entrypoint with adaptive batching to mitigate OOM.

    Accepts PointCloud containing xyz coordinates.
    `cull_func` filters projected points using the provided bounds mask.
    """
    assert isinstance(pc, PointCloud), f"{type(pc)=}"
    assert isinstance(camera, Camera), f"{type(camera)=}"
    points = pc.xyz

    camera_prepared = camera.to(
        device=points.device, convention="opencv"
    ).scale_intrinsics(resolution=resolution)
    intrinsics = camera_prepared.intrinsics
    extrinsics = camera_prepared.extrinsics

    # If `num_divide` is set, derive batch size from N / 2**num_divide.
    N = points.shape[0]
    if num_divide is not None:
        bs = max(1, math.ceil(N / (2**num_divide)))
        return _prepare_points_for_rendering_batched(
            points=points,
            camera=camera_prepared,
            resolution=resolution,
            batch_size=bs,
            cull_func=cull_func,
        )

    # Otherwise, progressively halve batch size on CUDA OOM up to `max_divide`.
    n = 0
    while n <= max_divide:
        bs = max(1, math.ceil(N / (2**n)))
        try:
            return _prepare_points_for_rendering_batched(
                points=points,
                camera=camera_prepared,
                resolution=resolution,
                batch_size=bs,
                cull_func=cull_func,
            )
        except torch.cuda.OutOfMemoryError:
            n += 1
            torch.cuda.empty_cache()
            continue
        except Exception:
            raise

    raise torch.cuda.OutOfMemoryError(
        f"CUDA OOM after {max_divide} divisions in prepare_points_for_rendering."
    )
