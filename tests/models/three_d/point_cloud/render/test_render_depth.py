"""Test cases for depth rendering from point clouds."""

from typing import List, Tuple

import pytest
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from models.three_d.point_cloud.render import (
    render_depth_from_point_cloud,
)


def test_render_depth_basic() -> None:
    """A depth map asked for without a mask comes back at the requested resolution, in float32, with every rendered pixel at a positive depth.

    Args:
        None.

    Returns:
        None.
    """
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],  # Center, depth 1
                [0.5, 0.5, -2.0],  # Upper right, depth 2
                [-0.5, 0.5, -1.5],  # Upper left, depth 1.5
                [0.0, -0.5, -3.0],  # Bottom center, depth 3
            ],
            dtype=torch.float32,
        )
    )

    camera = _build_camera(focal=100.0, principal_point=50.0)

    depth_map = render_depth_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
        return_mask=False,
    )

    assert depth_map.shape == (100, 100) and depth_map.dtype == torch.float32, (
        "The depth map must be [100, 100] and float32. "
        f"{depth_map.shape=} {depth_map.dtype=}"
    )
    assert (depth_map[depth_map != -1.0] > 0).all(), (
        "Every depth other than the -1.0 background must be positive. "
        f"{depth_map[depth_map != -1.0].min()=}"
    )


def test_render_depth_with_mask() -> None:
    """The mask a caller asks for marks exactly the rendered pixels, and the background carries the ignore value on the rest.

    Args:
        None.

    Returns:
        None.
    """
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.2, 0.2, -1.5],
                [-0.3, -0.3, -2.0],
            ],
            dtype=torch.float32,
        )
    )

    camera = _build_camera(focal=100.0, principal_point=50.0)

    depth_map, valid_mask = render_depth_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
        return_mask=True,
    )

    assert (
        depth_map.shape == (100, 100)
        and valid_mask.shape == (100, 100)
        and valid_mask.dtype == torch.bool
    ), (
        "Both maps must be [100, 100] and the mask must be bool. "
        f"{depth_map.shape=} {valid_mask.shape=} {valid_mask.dtype=}"
    )
    assert valid_mask.sum() > 0 and valid_mask.sum() < 100 * 100, (
        "The mask must cover some pixels but not the whole image. "
        f"{valid_mask.sum()=}"
    )
    assert (depth_map[valid_mask] > 0).all() and (
        depth_map[~valid_mask] == -1.0
    ).all(), (
        "Depths must be positive under the mask and -1.0 outside it. "
        f"{depth_map[valid_mask].min()=} {depth_map[~valid_mask].unique()=}"
    )


def test_render_depth_sorting() -> None:
    """Two points on one ray render as the near one, since a farther point must not overwrite what occludes it.

    Args:
        None.

    Returns:
        None.
    """
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -3.0],  # Farther point (depth 3)
                [0.0, 0.0, -1.0],  # Closer point (depth 1, should overwrite)
            ],
            dtype=torch.float32,
        )
    )

    camera = _build_camera(focal=100.0, principal_point=50.0)

    depth_map = render_depth_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
    )

    valid_depths = depth_map[depth_map != -1.0]
    if len(valid_depths) > 0:
        assert valid_depths.min() < 1.5, (
            "The ray's two points must render as the near one. "
            f"{valid_depths.min()=} {valid_depths.unique()=}"
        )


def test_render_depth_custom_ignore_value() -> None:
    """The background value is the caller's to name, and it reaches every pixel no point projected onto.

    Args:
        None.

    Returns:
        None.
    """
    pc_data = PointCloud(xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32))

    camera = _build_camera(focal=100.0, principal_point=50.0)

    depth_map = render_depth_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
        ignore_value=-999.0,
    )

    assert (depth_map == -999.0).sum() > 100 * 100 * 0.9, (
        "Over nine tenths of the pixels must carry the -999.0 ignore value. "
        f"{(depth_map == -999.0).sum()=}"
    )


def test_render_depth_points_behind_camera() -> None:
    """A point behind an OpenGL camera is dropped rather than folded back in front of it.

    Args:
        None.

    Returns:
        None.
    """
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],  # Behind camera (positive Z in OpenGL)
                [0.0, 0.0, -1.0],  # In front of camera
            ],
            dtype=torch.float32,
        )
    )

    camera = _build_camera(focal=100.0, principal_point=50.0)

    depth_map, valid_mask = render_depth_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
        return_mask=True,
    )

    assert valid_mask.sum() >= 1 and (depth_map[valid_mask] > 0).all(), (
        "The mask must cover at least one pixel and every depth under it must be positive. "
        f"{valid_mask.sum()=} {depth_map[valid_mask].tolist()=}"
    )


def test_render_depth_multiple_points_per_pixel() -> None:
    """Several points landing on one pixel resolve to the nearest, which is the same rule sorting states at pixel granularity.

    Args:
        None.

    Returns:
        None.
    """
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.001, 0.0, -2.0],
                [0.0, -0.001, -1.5],
                [0.001, -0.001, -3.0],
            ],
            dtype=torch.float32,
        )
    )

    camera = _build_camera(focal=1000.0, principal_point=50.0)

    depth_map = render_depth_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
    )

    valid_depths = depth_map[48:52, 48:52][depth_map[48:52, 48:52] != -1.0]
    if len(valid_depths) > 0:
        assert valid_depths.min() < 1.5, (
            "The points sharing the central pixels must resolve to the nearest. "
            f"{valid_depths.min()=} {valid_depths.unique()=}"
        )


def test_render_depth_intrinsics_scaling() -> None:
    """One camera renders at two resolutions, so the intrinsics are scaled to the request rather than pinned to the camera's own extents.

    Args:
        None.

    Returns:
        None.
    """
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.5, 0.5, -2.0],
            ],
            dtype=torch.float32,
        )
    )

    camera = _build_camera(focal=100.0, principal_point=50.0)

    depth_map_small = render_depth_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(50, 50),
    )

    depth_map_large = render_depth_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(200, 200),
    )

    assert depth_map_small.shape == (50, 50) and depth_map_large.shape == (200, 200), (
        "Each map must have the resolution it was asked for. "
        f"{depth_map_small.shape=} {depth_map_large.shape=}"
    )
    assert (depth_map_small != -1.0).any() and (depth_map_large != -1.0).any(), (
        "Each map must carry at least one rendered pixel. "
        f"{(depth_map_small != -1.0).sum()=} {(depth_map_large != -1.0).sum()=}"
    )


def test_render_depth_batched_matches_per_camera() -> None:
    """A Cameras of several poses renders one cloud in a single call to [B, H, W], each slice equal to what that pose renders on its own.

    Args:
        None.

    Returns:
        None.
    """
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.1, 0.1, -2.0],
                [-0.1, 0.1, -1.5],
                [0.0, -0.1, -3.0],
            ],
            dtype=torch.float32,
        )
    )

    cameras = _build_cameras(
        focal=100.0,
        principal_point=50.0,
        translations=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.15, 0.0)],
    )

    depth_maps = render_depth_from_point_cloud(
        pc=pc_data,
        camera=cameras,
        resolution=(64, 80),
    )

    assert depth_maps.shape == (3, 64, 80) and depth_maps.dtype == torch.float32, (
        "The batched depth map must be [3, 64, 80] and float32. "
        f"{depth_maps.shape=} {depth_maps.dtype=}"
    )

    for index, camera in enumerate(cameras):
        depth_map = render_depth_from_point_cloud(
            pc=pc_data,
            camera=camera,
            resolution=(64, 80),
        )
        assert torch.equal(depth_maps[index], depth_map), (
            "Each batched slice must equal what its own camera renders alone. "
            f"{index=} {(depth_maps[index] - depth_map).abs().max()=} "
            f"{(depth_maps[index] != depth_map).sum()=}"
        )


def test_render_depth_batch_of_one_keeps_its_axis() -> None:
    """A Cameras of length one renders to [1, H, W] rather than [H, W].

    Args:
        None.

    Returns:
        None.
    """
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.1, 0.1, -2.0],
                [-0.1, 0.1, -1.5],
                [0.0, -0.1, -3.0],
            ],
            dtype=torch.float32,
        )
    )

    cameras = _build_cameras(
        focal=100.0,
        principal_point=50.0,
        translations=[(0.0, 0.0, 0.0)],
    )

    depth_maps = render_depth_from_point_cloud(
        pc=pc_data,
        camera=cameras,
        resolution=(64, 80),
    )

    assert depth_maps.shape == (1, 64, 80), (
        "A Cameras of length one must render to [1, 64, 80], keeping its leading axis. "
        f"{depth_maps.shape=}"
    )

    depth_map = render_depth_from_point_cloud(
        pc=pc_data,
        camera=next(iter(cameras)),
        resolution=(64, 80),
    )

    assert depth_map.shape == (64, 80) and torch.equal(depth_maps[0], depth_map), (
        "The one Camera's map must be [64, 80] and equal the batched map's only slice. "
        f"{depth_map.shape=} {depth_maps.shape=} "
        f"{(depth_map != -1.0).sum()=} {(depth_maps[0] != -1.0).sum()=}"
    )


def test_render_depth_batched_cull_is_per_camera() -> None:
    """Cameras seeing different subsets of one cloud each keep their own survivors, culling marking a per-camera mask rather than compacting.

    Args:
        None.

    Returns:
        None.
    """
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.2, -1.0],  # Inside camera 0's bounds, outside camera 1's
                [2.0, -0.2, -1.0],  # Inside camera 1's bounds, outside camera 0's
            ],
            dtype=torch.float32,
        )
    )

    cameras = _build_cameras(
        focal=100.0,
        principal_point=50.0,
        translations=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
    )

    depth_maps, valid_masks = render_depth_from_point_cloud(
        pc=pc_data,
        camera=cameras,
        resolution=(64, 80),
        return_mask=True,
    )

    assert (
        depth_maps.shape == (2, 64, 80)
        and valid_masks.shape == (2, 64, 80)
        and valid_masks.dtype == torch.bool
    ), (
        "Both batched maps must be [2, 64, 80] and the mask must be bool. "
        f"{depth_maps.shape=} {valid_masks.shape=} {valid_masks.dtype=}"
    )
    assert (valid_masks[0] & ~valid_masks[1]).any() and (
        valid_masks[1] & ~valid_masks[0]
    ).any(), (
        "Each camera's mask must cover pixels the other camera's mask does not. "
        f"{(valid_masks[0] & ~valid_masks[1]).sum()=} "
        f"{(valid_masks[1] & ~valid_masks[0]).sum()=}"
    )


def test_render_depth_occlusion_holds_when_pixels_collide() -> None:
    """A cloud dense enough that many points share a pixel still renders the nearest of them, and renders the same map every run, since occlusion must not depend on which write landed last.

    Args:
        None.

    Returns:
        None.
    """
    focal = 100.0
    principal_point = 50.0
    resolution = (32, 32)
    render_height, render_width = resolution

    # The camera's own extents are twice its principal point, so rendering at this resolution restates its intrinsics by that ratio.
    render_fx = focal * render_width / (2.0 * principal_point)
    render_fy = focal * render_height / (2.0 * principal_point)
    render_cx = principal_point * render_width / (2.0 * principal_point)
    render_cy = principal_point * render_height / (2.0 * principal_point)

    # Several thousand points into 1024 pixels, each aimed at the centre of a uniformly drawn pixel so most pixels take several of them.
    num_points = 4096
    generator = torch.Generator().manual_seed(0)
    target_columns = torch.randint(
        low=0, high=render_width, size=(num_points,), generator=generator
    )
    target_rows = torch.randint(
        low=0, high=render_height, size=(num_points,), generator=generator
    )
    depths = 1.0 + 3.0 * torch.rand(num_points, generator=generator)
    pc_data = PointCloud(
        xyz=torch.stack(
            [
                (target_columns + 0.5 - render_cx) * depths / render_fx,
                -(target_rows + 0.5 - render_cy) * depths / render_fy,
                -depths,
            ],
            dim=1,
        )
    )

    camera = _build_camera(focal=focal, principal_point=principal_point)

    depth_maps = []
    for _ in range(6):
        depth_maps.append(
            render_depth_from_point_cloud(
                pc=pc_data,
                camera=camera,
                resolution=resolution,
            )
        )

    for render_index, depth_map in enumerate(depth_maps):
        assert torch.equal(depth_map, depth_maps[0]), (
            "Repeated renders of one camera must give the same depth map. "
            f"{render_index=} {(depth_map - depth_maps[0]).abs().max()=}"
        )

    # The nearest depth per pixel, projected here rather than read back from the render.
    point_depths = -pc_data.xyz[:, 2]
    point_columns = pc_data.xyz[:, 0] / point_depths * render_fx + render_cx
    point_rows = -pc_data.xyz[:, 1] / point_depths * render_fy + render_cy
    inside = (
        (point_depths > 0)
        & (point_columns >= 0)
        & (point_columns < render_width)
        & (point_rows >= 0)
        & (point_rows < render_height)
    )
    expected_depth_map = torch.full(resolution, -1.0, dtype=torch.float32)
    points_per_pixel = torch.zeros(resolution, dtype=torch.int64)
    for point_row, point_column, point_depth in zip(
        point_rows[inside].long().tolist(),
        point_columns[inside].long().tolist(),
        point_depths[inside].tolist(),
        strict=True,
    ):
        if (
            points_per_pixel[point_row, point_column] == 0
            or point_depth < expected_depth_map[point_row, point_column]
        ):
            expected_depth_map[point_row, point_column] = point_depth
        points_per_pixel[point_row, point_column] += 1

    assert torch.equal(depth_maps[0], expected_depth_map), (
        "Each rendered pixel must carry the smallest depth among the points that "
        "projected onto it. "
        f"{(depth_maps[0] - expected_depth_map).abs().max()=} "
        f"{(depth_maps[0] != expected_depth_map).sum()=}"
    )


def test_render_depth_batched_matches_per_camera_when_pixels_collide() -> None:
    """The per-camera equality the batch promises holds at that same density, which is the regime a batch is rendered at.

    Args:
        None.

    Returns:
        None.
    """
    focal = 100.0
    principal_point = 50.0
    resolution = (32, 32)
    render_height, render_width = resolution

    render_fx = focal * render_width / (2.0 * principal_point)
    render_fy = focal * render_height / (2.0 * principal_point)
    render_cx = principal_point * render_width / (2.0 * principal_point)
    render_cy = principal_point * render_height / (2.0 * principal_point)

    num_points = 4096
    generator = torch.Generator().manual_seed(0)
    target_columns = torch.randint(
        low=0, high=render_width, size=(num_points,), generator=generator
    )
    target_rows = torch.randint(
        low=0, high=render_height, size=(num_points,), generator=generator
    )
    depths = 1.0 + 3.0 * torch.rand(num_points, generator=generator)
    pc_data = PointCloud(
        xyz=torch.stack(
            [
                (target_columns + 0.5 - render_cx) * depths / render_fx,
                -(target_rows + 0.5 - render_cy) * depths / render_fy,
                -depths,
            ],
            dim=1,
        )
    )

    cameras = _build_cameras(
        focal=focal,
        principal_point=principal_point,
        translations=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.15, 0.0)],
    )

    depth_maps = render_depth_from_point_cloud(
        pc=pc_data,
        camera=cameras,
        resolution=resolution,
    )

    for index, camera in enumerate(cameras):
        depth_map = render_depth_from_point_cloud(
            pc=pc_data,
            camera=camera,
            resolution=resolution,
        )
        assert torch.equal(depth_maps[index], depth_map), (
            "Each batched slice must equal what its own camera renders alone. "
            f"{index=} {(depth_maps[index] - depth_map).abs().max()=} "
            f"{(depth_maps[index] != depth_map).sum()=}"
        )


def test_render_depth_point_size_dilates_the_rendered_discs() -> None:
    """A point size above one pixel grows each rendered point into a disc, so the parameter the entry point takes changes what it returns.

    Args:
        None.

    Returns:
        None.
    """
    pc_data = PointCloud(xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32))

    camera = _build_camera(focal=100.0, principal_point=50.0)

    _, valid_mask_narrow = render_depth_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
        return_mask=True,
        point_size=1.0,
    )
    depth_map_wide, valid_mask_wide = render_depth_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
        return_mask=True,
        point_size=5.0,
    )

    assert (
        valid_mask_wide.sum() > valid_mask_narrow.sum()
        and (depth_map_wide[valid_mask_wide] == 1.0).all()
    ), (
        "A wider point size must cover strictly more pixels, all at that point's own depth. "
        f"{valid_mask_narrow.sum()=} {valid_mask_wide.sum()=} "
        f"{depth_map_wide[valid_mask_wide].unique()=}"
    )


def test_render_depth_point_size_keeps_a_nan_background() -> None:
    """A NaN background survives the dilation: the discs cover the pixels they cover under a finite background, and every pixel outside them stays NaN.

    Args:
        None.

    Returns:
        None.
    """
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],  # Center, depth 1
                [0.5, 0.5, -2.0],  # Upper right, depth 2
                [-0.5, 0.5, -1.5],  # Upper left, depth 1.5
                [0.0, -0.5, -3.0],  # Bottom center, depth 3
            ],
            dtype=torch.float32,
        )
    )

    camera = _build_camera(focal=100.0, principal_point=50.0)

    depth_map_nan, valid_mask_nan = render_depth_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
        ignore_value=float('nan'),
        return_mask=True,
        point_size=3.0,
    )
    depth_map_finite, valid_mask_finite = render_depth_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
        ignore_value=-1.0,
        return_mask=True,
        point_size=3.0,
    )

    assert torch.equal(valid_mask_nan, valid_mask_finite), (
        "A NaN background must leave the discs covering the same pixels a finite background does. "
        f"{valid_mask_nan.sum()=} {valid_mask_finite.sum()=} "
        f"{(valid_mask_nan != valid_mask_finite).sum()=}"
    )
    assert torch.equal(torch.isnan(depth_map_nan), ~valid_mask_nan), (
        "The NaN-background map must be NaN exactly at the pixels its mask leaves uncovered. "
        f"{torch.isnan(depth_map_nan).sum()=} {(~valid_mask_nan).sum()=}"
    )
    assert torch.equal(
        depth_map_nan[valid_mask_nan], depth_map_finite[valid_mask_nan]
    ), (
        "The two maps must carry the same depth at every covered pixel. "
        f"{(depth_map_nan[valid_mask_nan] != depth_map_finite[valid_mask_nan]).sum()=}"
    )


def test_render_depth_invalid_inputs() -> None:
    """The malformed inputs are refused where each is first named, which for the two camera pieces is their own construction rather than the render call.

    Args:
        None.

    Returns:
        None.
    """
    valid_pc_data = PointCloud(
        xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    )
    valid_camera = _build_camera(focal=100.0, principal_point=50.0)

    with pytest.raises(AssertionError):
        render_depth_from_point_cloud(
            pc="not a point cloud",
            camera=valid_camera,
            resolution=(100, 100),
        )

    # A non-CameraIntrinsics intrinsics is rejected at Camera construction.
    with pytest.raises(AssertionError):
        Camera(
            intrinsics=torch.eye(4, dtype=torch.float32),
            extrinsics=valid_camera.extrinsics,
            device=torch.device("cpu"),
        )

    # A malformed (3x3) extrinsics matrix is rejected at CameraExtrinsics construction.
    with pytest.raises(AssertionError):
        CameraExtrinsics(
            extrinsics=torch.eye(3, dtype=torch.float32),
            extr_convention="opengl",
            device=torch.device("cpu"),
        )

    with pytest.raises(AssertionError):
        render_depth_from_point_cloud(
            pc=valid_pc_data,
            camera=valid_camera,
            resolution=(0, 100),
        )


def _build_camera(focal: float, principal_point: float) -> Camera:
    """Builds the identity-pose OpenGL pinhole camera on the CPU that every single-camera case here renders through.

    Args:
        focal: Shared focal length used for both fx and fy.
        principal_point: Shared principal-point coordinate used for both cx and cy.

    Returns:
        A Camera whose pinhole intrinsics are (fx, fy, cx, cy) and whose extrinsics are the identity cam2world matrix in the opengl convention.
    """
    return Camera(
        intrinsics=build_camera_intrinsics(
            model="pinhole",
            params={
                "fx": focal,
                "fy": focal,
                "cx": principal_point,
                "cy": principal_point,
                "h": int(round(2.0 * principal_point)),
                "w": int(round(2.0 * principal_point)),
            },
            intr_convention="standard",
            device=torch.device("cpu"),
        ),
        extrinsics=CameraExtrinsics(
            extrinsics=torch.eye(4, dtype=torch.float32),
            extr_convention="opengl",
            device=torch.device("cpu"),
        ),
        device=torch.device("cpu"),
    )


def _build_cameras(
    focal: float,
    principal_point: float,
    translations: List[Tuple[float, float, float]],
) -> Cameras:
    """Builds the OpenGL pinhole batch every batched case here renders through, one pose per translation.

    Args:
        focal: Shared focal length used for both fx and fy of every camera.
        principal_point: Shared principal-point coordinate used for both cx and cy of every camera.
        translations: Per-camera (x, y, z) world-space camera positions, whose length is the batch size B.

    Returns:
        A Cameras whose pinhole intrinsics params are each a [B] torch.Tensor of (fx, fy, cx, cy) and whose extrinsics are a [B, 4, 4] float32 stack of identity cam2world matrices carrying one translation each, in the opengl convention.
    """
    batch_size = len(translations)
    extrinsics = torch.eye(4, dtype=torch.float32).repeat(batch_size, 1, 1)
    extrinsics[:, :3, 3] = torch.tensor(translations, dtype=torch.float32)
    return Cameras(
        intrinsics=build_camera_intrinsics(
            model="pinhole",
            params={
                "fx": torch.full((batch_size,), focal),
                "fy": torch.full((batch_size,), focal),
                "cx": torch.full((batch_size,), principal_point),
                "cy": torch.full((batch_size,), principal_point),
                "h": torch.full((batch_size,), 2.0 * principal_point),
                "w": torch.full((batch_size,), 2.0 * principal_point),
            },
            intr_convention="standard",
            device=torch.device("cpu"),
        ),
        extrinsics=CameraExtrinsics(
            extrinsics=extrinsics,
            extr_convention="opengl",
            device=torch.device("cpu"),
        ),
        device=torch.device("cpu"),
    )
