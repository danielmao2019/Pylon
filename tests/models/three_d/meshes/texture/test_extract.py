"""Tests for seam-safe UV rasterization helpers."""

from typing import Any, Dict

import pytest
import torch

import models.three_d.meshes.texture.extract.extract as extract_module
from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.mesh.mesh import Mesh
from models.three_d.meshes.texture.extract import (
    _build_camera_uv_interpolation_data,
    _build_uv_rasterization_data,
    _build_uv_rasterization_mesh,
    compute_f_visibility_mask,
    extract_texture_from_images,
)
from models.three_d.meshes.texture.extract.visibility.texel_visibility import (
    _compute_visible_uv_texels_from_uv_polygon_regions,
    _geom,
    _map_visible_screen_space_polygon_regions_to_uv,
)


def test_build_uv_rasterization_mesh_duplicates_seam_crossing_face() -> None:
    """Split a cylindrical seam face into two UV triangles."""

    vertex_uv = torch.tensor(
        [
            [0.99, 0.20],
            [0.01, 0.25],
            [0.02, 0.80],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

    uv_rasterization_mesh = _build_uv_rasterization_mesh(
        vertex_uv=vertex_uv,
        faces=faces,
    )

    assert torch.equal(
        uv_rasterization_mesh["tri_i32"],
        torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32),
    ), f"{uv_rasterization_mesh['tri_i32']=}"
    assert torch.equal(
        uv_rasterization_mesh["raster_vertex_indices"],
        torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long),
    ), f"{uv_rasterization_mesh['raster_vertex_indices']=}"
    assert torch.equal(
        uv_rasterization_mesh["raster_face_indices"],
        torch.tensor([0, 0], dtype=torch.long),
    ), f"{uv_rasterization_mesh['raster_face_indices']=}"
    assert torch.allclose(
        uv_rasterization_mesh["raster_vertex_uv"][:3],
        torch.tensor(
            [
                [0.99, 0.20],
                [1.01, 0.25],
                [1.02, 0.80],
            ],
            dtype=torch.float32,
        ),
    ), f"{uv_rasterization_mesh['raster_vertex_uv']=}"
    assert torch.allclose(
        uv_rasterization_mesh["raster_vertex_uv"][3:],
        torch.tensor(
            [
                [-0.01, 0.20],
                [0.01, 0.25],
                [0.02, 0.80],
            ],
            dtype=torch.float32,
        ),
    ), f"{uv_rasterization_mesh['raster_vertex_uv']=}"


def test_build_uv_rasterization_mesh_keeps_non_seam_face_single() -> None:
    """Leave a non-seam face as one UV triangle."""

    vertex_uv = torch.tensor(
        [
            [0.20, 0.10],
            [0.30, 0.15],
            [0.25, 0.90],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

    uv_rasterization_mesh = _build_uv_rasterization_mesh(
        vertex_uv=vertex_uv,
        faces=faces,
    )

    assert torch.equal(
        uv_rasterization_mesh["tri_i32"],
        torch.tensor([[0, 1, 2]], dtype=torch.int32),
    ), f"{uv_rasterization_mesh['tri_i32']=}"
    assert torch.equal(
        uv_rasterization_mesh["raster_vertex_indices"],
        torch.tensor([0, 1, 2], dtype=torch.long),
    ), f"{uv_rasterization_mesh['raster_vertex_indices']=}"
    assert torch.equal(
        uv_rasterization_mesh["raster_face_indices"],
        torch.tensor([0], dtype=torch.long),
    ), f"{uv_rasterization_mesh['raster_face_indices']=}"
    assert torch.allclose(
        uv_rasterization_mesh["raster_vertex_uv"],
        vertex_uv,
    ), f"{uv_rasterization_mesh['raster_vertex_uv']=} {vertex_uv=}"


def test_build_camera_uv_interpolation_data_shifts_seam_face_once() -> None:
    """Shift one seam face into one continuous UV chart for camera interpolation.

    Args:
        None.

    Returns:
        None.
    """

    vertex_uv = torch.tensor(
        [
            [0.99, 0.20],
            [0.01, 0.25],
            [0.02, 0.80],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

    camera_uv_interpolation_data = _build_camera_uv_interpolation_data(
        vertex_uv=vertex_uv,
        faces=faces,
    )

    assert torch.equal(
        camera_uv_interpolation_data["camera_attr_tri_i32"],
        torch.tensor([[0, 1, 2]], dtype=torch.int32),
    ), f"{camera_uv_interpolation_data['camera_attr_tri_i32']=}"
    assert torch.allclose(
        camera_uv_interpolation_data["camera_attr_vertex_uv"],
        torch.tensor(
            [
                [0.99, 0.20],
                [1.01, 0.25],
                [1.02, 0.80],
            ],
            dtype=torch.float32,
        ),
    ), f"{camera_uv_interpolation_data['camera_attr_vertex_uv']=}"


def test_vertex_uv_to_clip_uses_rasterizer_buffer_v_mapping() -> None:
    """Map small-`v` UV coordinates to negative clip-space `y`.

    Args:
        None.

    Returns:
        None.
    """

    vertex_uv = torch.tensor(
        [
            [0.25, 0.00],
            [0.75, 1.00],
        ],
        dtype=torch.float32,
    )

    uv_clip = extract_module._vertex_uv_to_clip(vertex_uv=vertex_uv)

    assert uv_clip.shape == (1, 2, 4), f"{uv_clip.shape=}"
    assert torch.allclose(
        uv_clip[0, :, :2],
        torch.tensor(
            [
                [-0.50, -1.00],
                [0.50, 1.00],
            ],
            dtype=torch.float32,
        ),
        atol=1.0e-6,
        rtol=0.0,
    ), f"{uv_clip=}"


def test_compute_f_visibility_mask_keeps_uv_channel_dimension() -> None:
    """Keep UV visibility masks in `[1, T, T, 1]` layout.

    Args:
        None.

    Returns:
        None.
    """

    vertices = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    cameras = Cameras(
        intrinsics=[torch.eye(3, dtype=torch.float32)],
        extrinsics=[torch.eye(4, dtype=torch.float32)],
        conventions=["opencv"],
        device="cpu",
    )
    uv_rasterization_data = {
        "uv_mask": torch.ones((1, 2, 2, 1), dtype=torch.float32),
        "camera_attr_vertex_uv": torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "camera_attr_tri_i32": torch.tensor([[0, 1, 2]], dtype=torch.int32),
    }

    visibility_mask = compute_f_visibility_mask(
        vertices=vertices,
        faces=faces,
        camera=cameras,
        image_height=2,
        image_width=2,
        uv_rasterization_data=uv_rasterization_data,
    )

    assert visibility_mask.shape == (1, 2, 2, 1), f"{visibility_mask.shape=}"


def test_compute_f_visibility_mask_uses_exact_camera_pixel_footprints() -> None:
    """Use exact camera-pixel footprints on a one-pixel image."""

    vertices = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    cameras = Cameras(
        intrinsics=[torch.eye(3, dtype=torch.float32)],
        extrinsics=[torch.eye(4, dtype=torch.float32)],
        conventions=["opencv"],
        device="cpu",
    )
    uv_rasterization_data = {
        "uv_mask": torch.ones((1, 2, 2, 1), dtype=torch.float32),
        "camera_attr_vertex_uv": torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "camera_attr_tri_i32": torch.tensor([[0, 1, 2]], dtype=torch.int32),
    }

    visibility_mask = compute_f_visibility_mask(
        vertices=vertices,
        faces=faces,
        camera=cameras,
        image_height=1,
        image_width=1,
        uv_rasterization_data=uv_rasterization_data,
    )

    assert visibility_mask.shape == (1, 2, 2, 1), f"{visibility_mask.shape=}"
    assert torch.any(visibility_mask > 0.0), f"{visibility_mask=}"
    assert torch.any(visibility_mask > 0.0), f"{visibility_mask=}"


def test_map_visible_screen_space_polygon_regions_to_uv_preserves_identity_face() -> (
    None
):
    """Map one unit-depth polygon to the same UV coordinates on an identity face.

    Args:
        None.

    Returns:
        None.
    """

    visible_screen_polygon_vertices = torch.tensor(
        [
            [
                [0.10, 0.10],
                [0.40, 0.10],
                [0.25, 0.25],
                [0.10, 0.20],
            ],
        ],
        dtype=torch.float32,
    )
    visible_screen_polygon_vertex_counts = torch.tensor([4], dtype=torch.long)
    visible_screen_polygon_face_indices = torch.tensor([0], dtype=torch.long)
    face_screen_vertices = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        ],
        dtype=torch.float32,
    )
    face_vertex_depth = torch.ones((1, 3), dtype=torch.float32)
    face_vertex_uv = face_screen_vertices.clone()

    uv_polygon_vertices, uv_polygon_vertex_counts = (
        _map_visible_screen_space_polygon_regions_to_uv(
            visible_screen_polygon_vertices=visible_screen_polygon_vertices,
            visible_screen_polygon_vertex_counts=visible_screen_polygon_vertex_counts,
            visible_screen_polygon_face_indices=visible_screen_polygon_face_indices,
            face_screen_vertices=face_screen_vertices,
            face_vertex_depth=face_vertex_depth,
            face_vertex_uv=face_vertex_uv,
        )
    )

    assert torch.equal(
        uv_polygon_vertex_counts,
        visible_screen_polygon_vertex_counts,
    ), f"{uv_polygon_vertex_counts=} {visible_screen_polygon_vertex_counts=}"
    assert torch.allclose(
        uv_polygon_vertices[0, :4],
        visible_screen_polygon_vertices[0, :4],
        atol=1.0e-6,
    ), f"{uv_polygon_vertices=} {visible_screen_polygon_vertices=}"


def test_break_visible_uv_polygon_regions_into_triangles_triangulates_quad_fan() -> (
    None
):
    """Triangulate one convex quad into a two-triangle fan.

    Args:
        None.

    Returns:
        None.
    """

    uv_polygon_vertices = torch.tensor(
        [
            [
                [0.10, 0.10],
                [0.50, 0.10],
                [0.50, 0.40],
                [0.10, 0.40],
            ],
        ],
        dtype=torch.float32,
    )
    uv_polygon_vertex_counts = torch.tensor([4], dtype=torch.long)

    uv_triangles = _geom._triangulate_convex_uv_polygons(
        polygon_vertices=uv_polygon_vertices,
        polygon_vertex_counts=uv_polygon_vertex_counts,
    )

    expected_uv_triangles = torch.tensor(
        [
            [
                [0.10, 0.10],
                [0.50, 0.10],
                [0.50, 0.40],
            ],
            [
                [0.10, 0.10],
                [0.50, 0.40],
                [0.10, 0.40],
            ],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(
        uv_triangles,
        expected_uv_triangles,
        atol=1.0e-6,
    ), f"{uv_triangles=} {expected_uv_triangles=}"


def test_compute_visible_uv_texels_from_uv_polygon_regions_uses_top_down_v_convention() -> (
    None
):
    """Map small-`v` UV triangles into the top half of the texel raster.

    Args:
        None.

    Returns:
        None.
    """

    uv_polygon_vertices = torch.tensor(
        [
            [
                [0.20, 0.05],
                [0.80, 0.05],
                [0.50, 0.25],
            ],
        ],
        dtype=torch.float32,
    )
    texture_size = 64
    uv_polygon_vertex_counts = torch.tensor([3], dtype=torch.long)

    exact_uv_visible = _compute_visible_uv_texels_from_uv_polygon_regions(
        uv_polygon_vertices=uv_polygon_vertices,
        uv_polygon_vertex_counts=uv_polygon_vertex_counts,
        texture_size=texture_size,
    )
    covered_rows = torch.nonzero(
        exact_uv_visible[0, :, :, 0] > 0.0,
        as_tuple=False,
    )[:, 0]

    assert covered_rows.numel() > 0, f"{covered_rows=}"
    assert int(covered_rows.max().item()) < (texture_size // 2), (
        "Expected small-`v` standard UV coordinates to occupy the top half "
        "of the texel raster. "
        f"{int(covered_rows.max().item())=} {texture_size=}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
def test_compute_f_visibility_mask_recovers_standard_uv_face_near_v_zero() -> None:
    """Recover most occupied texels for one fully visible face with standard UVs.

    Args:
        None.

    Returns:
        None.
    """

    device = torch.device("cuda")
    vertices = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], device=device, dtype=torch.long)
    vertex_uv = torch.tensor(
        [
            [0.20, 0.05],
            [0.50, 0.25],
            [0.80, 0.05],
        ],
        device=device,
        dtype=torch.float32,
    )
    cameras = Cameras(
        intrinsics=[torch.eye(3, device=device, dtype=torch.float32)],
        extrinsics=[torch.eye(4, device=device, dtype=torch.float32)],
        conventions=["opencv"],
        device=device,
    )
    uv_rasterization_data = _build_uv_rasterization_data(
        vertices=vertices,
        vertex_uv=vertex_uv,
        faces=faces,
        texture_size=64,
    )

    visibility_mask = compute_f_visibility_mask(
        vertices=vertices,
        faces=faces,
        camera=cameras,
        image_height=8,
        image_width=8,
        uv_rasterization_data=uv_rasterization_data,
    )
    coverage_fraction = float(
        (
            visibility_mask.sum()
            / torch.clamp(uv_rasterization_data["uv_mask"].sum(), min=1.0)
        ).item()
    )

    assert coverage_fraction > 0.9, f"{coverage_fraction=}"


def test_extract_texture_from_images_reuses_single_mesh_across_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use one shared mesh for all views when a single mesh is provided.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """

    def _fake_extract_vertex_color_from_single_image(
        mesh: Mesh,
        image: torch.Tensor,
        camera: Cameras,
        weights_cfg: Dict[str, Any],
        default_color: float,
    ) -> Dict[str, torch.Tensor]:
        assert isinstance(mesh, Mesh), f"{type(mesh)=}"
        assert isinstance(image, torch.Tensor), f"{type(image)=}"
        assert isinstance(camera, Cameras), f"{type(camera)=}"
        assert isinstance(weights_cfg, dict), f"{type(weights_cfg)=}"
        assert isinstance(default_color, float), f"{type(default_color)=}"
        base_color = mesh.vertices[:, :1].repeat(1, 3)
        view_offset = float(image.mean().item())
        return {
            "texture": base_color + view_offset,
            "weight": torch.ones((mesh.vertices.shape[0], 1), dtype=torch.float32),
        }

    monkeypatch.setattr(
        extract_module,
        "_extract_vertex_color_from_single_image",
        _fake_extract_vertex_color_from_single_image,
    )

    mesh = Mesh(
        vertices=torch.tensor(
            [[0.10, 0.0, 0.0], [0.20, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 1]], dtype=torch.long),
    )
    images = torch.stack(
        [
            torch.zeros((3, 2, 2), dtype=torch.float32),
            torch.ones((3, 2, 2), dtype=torch.float32),
        ],
        dim=0,
    )
    cameras = Cameras(
        intrinsics=[
            torch.eye(3, dtype=torch.float32),
            torch.eye(3, dtype=torch.float32),
        ],
        extrinsics=[
            torch.eye(4, dtype=torch.float32),
            torch.eye(4, dtype=torch.float32),
        ],
        conventions=["opencv", "opencv"],
        device="cpu",
    )

    extracted_vertex_color = extract_texture_from_images(
        mesh=mesh,
        images=images,
        cameras=cameras,
        weights_cfg={"weights": "visible"},
    )

    expected_vertex_color = torch.tensor(
        [[0.60, 0.60, 0.60], [0.70, 0.70, 0.70]],
        dtype=torch.float32,
    )
    assert torch.allclose(
        extracted_vertex_color, expected_vertex_color
    ), f"{extracted_vertex_color=} {expected_vertex_color=}"


def test_extract_texture_from_images_uses_per_view_mesh_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use one mesh per view when multiple meshes are provided.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """

    def _fake_extract_vertex_color_from_single_image(
        mesh: Mesh,
        image: torch.Tensor,
        camera: Cameras,
        weights_cfg: Dict[str, Any],
        default_color: float,
    ) -> Dict[str, torch.Tensor]:
        assert isinstance(mesh, Mesh), f"{type(mesh)=}"
        assert isinstance(image, torch.Tensor), f"{type(image)=}"
        assert isinstance(camera, Cameras), f"{type(camera)=}"
        assert isinstance(weights_cfg, dict), f"{type(weights_cfg)=}"
        assert isinstance(default_color, float), f"{type(default_color)=}"
        return {
            "texture": mesh.vertices[:, :1].repeat(1, 3),
            "weight": torch.ones((mesh.vertices.shape[0], 1), dtype=torch.float32),
        }

    monkeypatch.setattr(
        extract_module,
        "_extract_vertex_color_from_single_image",
        _fake_extract_vertex_color_from_single_image,
    )

    meshes = [
        Mesh(
            vertices=torch.tensor(
                [[0.10, 0.0, 0.0], [0.20, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            faces=torch.tensor([[0, 1, 1]], dtype=torch.long),
        ),
        Mesh(
            vertices=torch.tensor(
                [[0.30, 0.0, 0.0], [0.40, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            faces=torch.tensor([[0, 1, 1]], dtype=torch.long),
        ),
    ]
    images = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
    cameras = Cameras(
        intrinsics=[
            torch.eye(3, dtype=torch.float32),
            torch.eye(3, dtype=torch.float32),
        ],
        extrinsics=[
            torch.eye(4, dtype=torch.float32),
            torch.eye(4, dtype=torch.float32),
        ],
        conventions=["opencv", "opencv"],
        device="cpu",
    )

    extracted_vertex_color = extract_texture_from_images(
        mesh=meshes,
        images=images,
        cameras=cameras,
        weights_cfg={"weights": "visible"},
    )

    expected_vertex_color = torch.tensor(
        [[0.20, 0.20, 0.20], [0.30, 0.30, 0.30]],
        dtype=torch.float32,
    )
    assert torch.allclose(
        extracted_vertex_color, expected_vertex_color
    ), f"{extracted_vertex_color=} {expected_vertex_color=}"


def test_extract_texture_from_images_rejects_per_view_mesh_count_mismatch() -> None:
    """Reject mesh-list inputs whose view count does not match images and cameras.

    Args:
        None.

    Returns:
        None.
    """

    mesh = Mesh(
        vertices=torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 1]], dtype=torch.long),
    )
    images = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
    cameras = Cameras(
        intrinsics=[
            torch.eye(3, dtype=torch.float32),
            torch.eye(3, dtype=torch.float32),
        ],
        extrinsics=[
            torch.eye(4, dtype=torch.float32),
            torch.eye(4, dtype=torch.float32),
        ],
        conventions=["opencv", "opencv"],
        device="cpu",
    )

    with pytest.raises(AssertionError):
        extract_texture_from_images(
            mesh=[mesh],
            images=images,
            cameras=cameras,
            weights_cfg={"weights": "visible"},
        )


def test_fuse_uv_texture_observations_returns_image_row_order() -> None:
    """Return fused UV outputs in ordinary image row order.

    Args:
        None.

    Returns:
        None.
    """

    observations = [
        {
            "texture": torch.tensor(
                [
                    [
                        [[0.10, 0.20, 0.30]],
                        [[0.90, 0.80, 0.70]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "weight": torch.tensor(
                [
                    [
                        [[1.0]],
                        [[0.0]],
                    ]
                ],
                dtype=torch.float32,
            ),
        }
    ]

    fused_outputs = extract_module._fuse_uv_texture_observations(
        observations=observations,
        weights_cfg={"weights": "visible"},
        default_color=0.7,
    )

    expected_texture = torch.tensor(
        [
            [
                [[0.10, 0.20, 0.30]],
                [[0.70, 0.70, 0.70]],
            ]
        ],
        dtype=torch.float32,
    )
    expected_valid_mask = torch.tensor(
        [
            [
                [[1.0]],
                [[0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(
        fused_outputs["texture"],
        expected_texture,
        atol=1.0e-6,
        rtol=0.0,
    ), f"{fused_outputs['texture']=} {expected_texture=}"
    assert torch.equal(
        fused_outputs["valid_mask"],
        expected_valid_mask,
    ), f"{fused_outputs['valid_mask']=} {expected_valid_mask=}"


def test_extract_uv_texture_map_from_single_image_returns_image_row_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return one-view UV observations in ordinary image row order.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """

    def _fake_extract_uv_texture_map_from_single_image(
        mesh: Mesh,
        image: torch.Tensor,
        camera: Cameras,
        weights_cfg: Dict[str, Any],
        uv_rasterization_data: Dict[str, torch.Tensor],
        polygon_rast_method: str,
    ) -> Dict[str, torch.Tensor]:
        assert isinstance(mesh, Mesh), f"{type(mesh)=}"
        assert isinstance(image, torch.Tensor), f"{type(image)=}"
        assert isinstance(camera, Cameras), f"{type(camera)=}"
        assert isinstance(weights_cfg, dict), f"{type(weights_cfg)=}"
        assert isinstance(
            uv_rasterization_data, dict
        ), f"{type(uv_rasterization_data)=}"
        assert isinstance(polygon_rast_method, str), f"{type(polygon_rast_method)=}"
        return {
            "texture": torch.tensor(
                [
                    [
                        [[0.90, 0.80, 0.70]],
                        [[0.10, 0.20, 0.30]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "weight": torch.tensor(
                [
                    [
                        [[0.0]],
                        [[1.0]],
                    ]
                ],
                dtype=torch.float32,
            ),
        }

    monkeypatch.setattr(
        extract_module,
        "_extract_uv_texture_map_from_single_image",
        _fake_extract_uv_texture_map_from_single_image,
    )

    mesh = Mesh(
        vertices=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.long),
        vertex_uv=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        face_uvs=torch.tensor([[0, 1, 2]], dtype=torch.long),
        convention="obj",
    )
    image = torch.zeros((3, 2, 2), dtype=torch.float32)
    camera = Cameras(
        intrinsics=[torch.eye(3, dtype=torch.float32)],
        extrinsics=[torch.eye(4, dtype=torch.float32)],
        conventions=["opencv"],
        device="cpu",
    )
    uv_rasterization_data = {"uv_mask": torch.ones((1, 2, 1, 1), dtype=torch.float32)}

    extracted_uv_texture_map = extract_module._extract_uv_texture_map_from_single_image(
        mesh=mesh,
        image=image,
        camera=camera,
        weights_cfg={"weights": "visible"},
        uv_rasterization_data=uv_rasterization_data,
        polygon_rast_method="v2",
    )

    expected_texture = torch.tensor(
        [
            [
                [[0.90, 0.80, 0.70]],
                [[0.10, 0.20, 0.30]],
            ]
        ],
        dtype=torch.float32,
    )
    expected_weight = torch.tensor(
        [
            [
                [[0.0]],
                [[1.0]],
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(
        extracted_uv_texture_map["texture"],
        expected_texture,
    ), f"{extracted_uv_texture_map['texture']=} {expected_texture=}"
    assert torch.equal(
        extracted_uv_texture_map["weight"],
        expected_weight,
    ), f"{extracted_uv_texture_map['weight']=} {expected_weight=}"


def test_extract_texture_from_images_keeps_uv_texture_row_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one-view UV extraction coherent through the public API.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """

    def _fake_build_uv_rasterization_data(
        vertices: torch.Tensor,
        vertex_uv: torch.Tensor,
        faces: torch.Tensor,
        texture_size: int,
        face_uvs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
        assert isinstance(vertex_uv, torch.Tensor), f"{type(vertex_uv)=}"
        assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
        assert isinstance(texture_size, int), f"{type(texture_size)=}"
        assert isinstance(face_uvs, torch.Tensor), f"{type(face_uvs)=}"
        return {"uv_mask": torch.ones((1, 2, 1, 1), dtype=torch.float32)}

    def _fake_extract_uv_texture_map_from_single_image(
        mesh: Mesh,
        image: torch.Tensor,
        camera: Cameras,
        weights_cfg: Dict[str, Any],
        uv_rasterization_data: Dict[str, torch.Tensor],
        polygon_rast_method: str,
    ) -> Dict[str, torch.Tensor]:
        assert isinstance(mesh, Mesh), f"{type(mesh)=}"
        assert isinstance(image, torch.Tensor), f"{type(image)=}"
        assert isinstance(camera, Cameras), f"{type(camera)=}"
        assert isinstance(weights_cfg, dict), f"{type(weights_cfg)=}"
        assert isinstance(
            uv_rasterization_data, dict
        ), f"{type(uv_rasterization_data)=}"
        assert isinstance(polygon_rast_method, str), f"{type(polygon_rast_method)=}"
        return {
            "texture": torch.tensor(
                [
                    [
                        [[0.10, 0.20, 0.30]],
                        [[0.90, 0.80, 0.70]],
                    ]
                ],
                dtype=torch.float32,
            ),
            "weight": torch.tensor(
                [
                    [
                        [[1.0]],
                        [[0.0]],
                    ]
                ],
                dtype=torch.float32,
            ),
        }

    monkeypatch.setattr(
        extract_module,
        "_build_uv_rasterization_data",
        _fake_build_uv_rasterization_data,
    )
    monkeypatch.setattr(
        extract_module,
        "_extract_uv_texture_map_from_single_image",
        _fake_extract_uv_texture_map_from_single_image,
    )

    mesh = Mesh(
        vertices=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.long),
        vertex_uv=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        face_uvs=torch.tensor([[0, 1, 2]], dtype=torch.long),
        convention="obj",
    )
    images = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    cameras = Cameras(
        intrinsics=[torch.eye(3, dtype=torch.float32)],
        extrinsics=[torch.eye(4, dtype=torch.float32)],
        conventions=["opencv"],
        device="cpu",
    )

    extracted_texture = extract_texture_from_images(
        mesh=mesh,
        images=images,
        cameras=cameras,
        weights_cfg={"weights": "visible"},
        texture_size=2,
        default_color=0.7,
    )

    expected_texture = torch.tensor(
        [
            [
                [[0.70, 0.70, 0.70]],
                [[0.10, 0.20, 0.30]],
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(
        extracted_texture,
        expected_texture,
    ), f"{extracted_texture=} {expected_texture=}"
