"""Focused regressions for the texel-center point-projection visibility path."""

from typing import Dict

import torch

from data.structures.three_d.camera.cameras import Cameras
from models.three_d.meshes.texture.extract.visibility.texel_visibility_v2 import (
    _compute_front_depth_gap_threshold_relative,
    _compute_texel_visibility_mask_from_world_coords,
    _select_visible_depth_clusters_per_camera_pixel,
    compute_f_visibility_mask_v2,
)


def _build_one_camera() -> Cameras:
    """Build one identity OpenCV camera for focused visibility tests.

    Args:
        None.

    Returns:
        One-camera batch on CPU.
    """

    return Cameras(
        intrinsics=[torch.eye(3, dtype=torch.float32)],
        extrinsics=[torch.eye(4, dtype=torch.float32)],
        conventions=["opencv"],
        device="cpu",
    )


def test_compute_f_visibility_mask_v2_maps_texel_centers_through_identity_face() -> (
    None
):
    """Keep the v2 texel-center pipeline consistent on one identity face.

    Args:
        None.

    Returns:
        None.
    """

    vertices = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 2, 1]], dtype=torch.long)
    cameras = _build_one_camera()
    uv_mask = torch.tensor(
        [
            [
                [[1.0], [1.0]],
                [[1.0], [0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    rast_out = torch.zeros((1, 2, 2, 4), dtype=torch.float32)
    rast_out[0, 0, 0, 3] = 1.0
    rast_out[0, 0, 1, 3] = 1.0
    rast_out[0, 1, 0, 3] = 1.0
    uv_rasterization_data: Dict[str, torch.Tensor] = {
        "uv_mask": uv_mask,
        "rast_out": rast_out,
        "raster_face_indices": torch.tensor([0], dtype=torch.long),
        "camera_attr_vertex_uv": torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    }

    visibility_mask = compute_f_visibility_mask_v2(
        vertices=vertices,
        faces=faces,
        camera=cameras,
        image_height=2,
        image_width=2,
        uv_rasterization_data=uv_rasterization_data,
    )

    assert visibility_mask.shape == (1, 2, 2, 1), f"{visibility_mask.shape=}"
    assert torch.equal(visibility_mask, uv_mask), (
        "Expected the three occupied texels to stay visible under the identity "
        "UV-to-world mapping with one texel per camera pixel. "
        f"{visibility_mask=} {uv_mask=}"
    )


def test_compute_f_visibility_mask_v2_filters_back_facing_face_texels() -> None:
    """Drop texels whose owning face is back-facing in the current camera view.

    Args:
        None.

    Returns:
        None.
    """

    vertices = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    cameras = _build_one_camera()
    uv_mask = torch.tensor(
        [
            [
                [[1.0], [1.0]],
                [[1.0], [0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    rast_out = torch.zeros((1, 2, 2, 4), dtype=torch.float32)
    rast_out[0, 0, 0, 3] = 1.0
    rast_out[0, 0, 1, 3] = 1.0
    rast_out[0, 1, 0, 3] = 1.0
    uv_rasterization_data: Dict[str, torch.Tensor] = {
        "uv_mask": uv_mask,
        "rast_out": rast_out,
        "raster_face_indices": torch.tensor([0], dtype=torch.long),
        "camera_attr_vertex_uv": torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    }

    visibility_mask = compute_f_visibility_mask_v2(
        vertices=vertices,
        faces=faces,
        camera=cameras,
        image_height=2,
        image_width=2,
        uv_rasterization_data=uv_rasterization_data,
    )

    expected_visibility_mask = torch.zeros_like(uv_mask)
    assert torch.equal(visibility_mask, expected_visibility_mask), (
        "Expected texels owned by a back-facing face to be removed before the "
        "camera-space visibility projection in v2. "
        f"{visibility_mask=} {expected_visibility_mask=}"
    )


def test_select_visible_depth_clusters_per_camera_pixel_stops_at_first_large_gap() -> (
    None
):
    """Keep only the front cluster when no later cluster is larger.

    Args:
        None.

    Returns:
        None.
    """

    linear_pixel_indices = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1],
        dtype=torch.long,
    )
    depth = torch.tensor(
        [
            1.0,
            1.0009765625,
            1.001953125,
            1.02,
            0.5,
            0.5009765625,
            0.501953125,
            0.53,
        ],
        dtype=torch.float32,
    )

    selection_mask = _select_visible_depth_clusters_per_camera_pixel(
        linear_pixel_indices=linear_pixel_indices,
        depth=depth,
        mesh_diagonal=1.0,
    )

    expected_selection_mask = torch.tensor(
        [True, True, True, False, True, True, True, False],
        dtype=torch.bool,
    )
    assert torch.equal(selection_mask, expected_selection_mask), (
        "Expected the selector to derive one frame-level threshold from the "
        "gap distribution, then keep the front contiguous depth prefix inside "
        "each camera pixel stack. "
        f"{selection_mask=} {expected_selection_mask=}"
    )


def test_select_visible_depth_clusters_per_camera_pixel_rejects_larger_second_cluster() -> (
    None
):
    """Reject a later cluster even when it is larger than the front sliver.

    Args:
        None.

    Returns:
        None.
    """

    linear_pixel_indices = torch.tensor(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        dtype=torch.long,
    )
    depth = torch.tensor(
        [
            1.0000000000,
            1.0009765625,
            1.0742187500,
            1.0751953125,
            1.0761718750,
            1.0771484375,
            0.5000000000,
            0.5009765625,
            0.5019531250,
            0.5300000000,
        ],
        dtype=torch.float32,
    )

    selection_mask = _select_visible_depth_clusters_per_camera_pixel(
        linear_pixel_indices=linear_pixel_indices,
        depth=depth,
        mesh_diagonal=1.0,
    )

    expected_selection_mask = torch.tensor(
        [True, True, False, False, False, False, True, True, True, False],
        dtype=torch.bool,
    )
    assert torch.equal(selection_mask, expected_selection_mask), (
        "Expected the selector to stop at the first large depth gap even when "
        "the later cluster is larger than the front sliver in the same "
        "camera-pixel stack. "
        f"{selection_mask=} {expected_selection_mask=}"
    )


def test_select_visible_depth_clusters_per_camera_pixel_rejects_smaller_second_cluster() -> (
    None
):
    """Reject a later cluster when it is smaller than the front cluster.

    Args:
        None.

    Returns:
        None.
    """

    linear_pixel_indices = torch.tensor(
        [0, 0, 0, 0, 0, 1, 1, 1, 1],
        dtype=torch.long,
    )
    depth = torch.tensor(
        [
            1.0000000000,
            1.0009765625,
            1.0019531250,
            1.0742187500,
            1.0751953125,
            0.5000000000,
            0.5009765625,
            0.5019531250,
            0.5300000000,
        ],
        dtype=torch.float32,
    )

    selection_mask = _select_visible_depth_clusters_per_camera_pixel(
        linear_pixel_indices=linear_pixel_indices,
        depth=depth,
        mesh_diagonal=1.0,
    )

    expected_selection_mask = torch.tensor(
        [True, True, True, False, False, True, True, True, False],
        dtype=torch.bool,
    )
    assert torch.equal(selection_mask, expected_selection_mask), (
        "Expected the selector to keep the old front-prefix behavior when the "
        "later cluster is smaller than the front cluster. "
        f"{selection_mask=} {expected_selection_mask=}"
    )


def test_select_visible_depth_clusters_per_camera_pixel_rejects_equal_second_cluster() -> (
    None
):
    """Reject a later cluster when it is only equal in size to the front cluster.

    Args:
        None.

    Returns:
        None.
    """

    linear_pixel_indices = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1],
        dtype=torch.long,
    )
    depth = torch.tensor(
        [
            1.0000000000,
            1.0009765625,
            1.0742187500,
            1.0751953125,
            0.5000000000,
            0.5009765625,
            0.5300000000,
            0.5600000000,
        ],
        dtype=torch.float32,
    )

    selection_mask = _select_visible_depth_clusters_per_camera_pixel(
        linear_pixel_indices=linear_pixel_indices,
        depth=depth,
        mesh_diagonal=1.0,
    )

    expected_selection_mask = torch.tensor(
        [True, True, False, False, True, True, False, False],
        dtype=torch.bool,
    )
    assert torch.equal(selection_mask, expected_selection_mask), (
        "Expected the selector to reject a later cluster when it is only equal "
        "in size to the front cluster, because the recovery rule stops at the "
        "first large gap regardless of later cluster size. "
        f"{selection_mask=} {expected_selection_mask=}"
    )


def test_compute_front_depth_gap_threshold_relative_splits_bimodal_gaps() -> None:
    """Derive a threshold between small intra-surface and large layer gaps.

    Args:
        None.

    Returns:
        None.
    """

    sorted_depth = torch.tensor(
        [
            1.00000,
            1.00010,
            1.01500,
            2.00000,
            2.00012,
            2.03000,
        ],
        dtype=torch.float32,
    )
    segment_start_mask = torch.tensor(
        [True, False, False, True, False, False],
        dtype=torch.bool,
    )

    threshold_relative = _compute_front_depth_gap_threshold_relative(
        sorted_depth=sorted_depth,
        segment_start_mask=segment_start_mask,
        mesh_diagonal=1.0,
    )

    assert 1.0e-4 < threshold_relative < 1.49e-2, (
        "Expected the distribution-driven threshold to fall between the small "
        "within-surface gaps and the large inter-layer gaps. "
        f"{threshold_relative=}"
    )


def test_compute_texel_visibility_mask_from_world_coords_keeps_front_depth_prefix() -> (
    None
):
    """Keep only the front depth prefix under the frame-level MAD threshold.

    Args:
        None.

    Returns:
        None.
    """

    world_coords = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0009765625],
            [0.0, 0.0, 1.001953125],
            [0.0, 0.0, 1.5],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0009765625],
            [1.0, 0.0, 1.001953125],
            [1.0, 0.0, 1.5],
        ],
        dtype=torch.float32,
    )
    valid_texel_indices = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
            [0, 6],
            [0, 7],
        ],
        dtype=torch.long,
    )
    valid_texel_mask = torch.tensor(
        [
            [
                [
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                ],
            ]
        ],
        dtype=torch.float32,
    )
    visibility_mask = _compute_texel_visibility_mask_from_world_coords(
        world_coords=world_coords,
        valid_texel_indices=valid_texel_indices,
        valid_texel_mask=valid_texel_mask,
        mesh_diagonal=1.0,
        camera=_build_one_camera(),
        image_height=1,
        image_width=2,
    )

    expected_visibility_mask = torch.tensor(
        [[[[1.0], [1.0], [1.0], [0.0], [1.0], [1.0], [1.0], [0.0]]]],
        dtype=torch.float32,
    )
    assert torch.equal(visibility_mask, expected_visibility_mask), (
        "Expected the v2 post-projection rescue to derive a frame-level gap "
        "threshold, then keep the front depth prefix in each occupied camera "
        "pixel stack. "
        f"{visibility_mask=} {expected_visibility_mask=}"
    )
