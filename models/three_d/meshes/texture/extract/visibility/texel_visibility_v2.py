"""Texel-visibility helpers based on UV-texel center point projection."""

from typing import Dict, Tuple

import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.point_cloud.camera.project import project_3d_to_2d
from data.structures.three_d.point_cloud.camera.transform import (
    world_to_camera_transform,
)
from models.three_d.meshes.texture.extract.normal_weights import (
    _compute_f_normals_weights,
)

FRONT_DEPTH_GAP_LOG_MAD_MULTIPLIER = 3.0

# -----------------------------------------------------------------------------
# Chunk 1: Main V2 Pipeline
# -----------------------------------------------------------------------------


def compute_f_visibility_mask_v2(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    camera: Cameras,
    image_height: int,
    image_width: int,
    uv_rasterization_data: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute one-view UV-pixel visibility mask from projected texel centers.

    Args:
        vertices: Mesh vertices `[V, 3]`.
        faces: Mesh faces `[F, 3]`.
        camera: One camera instance.
        image_height: Image height in pixels.
        image_width: Image width in pixels.
        uv_rasterization_data: Precomputed UV rasterization tensors.

    Returns:
        Float tensor `[1, T, T, 1]` with values in `{0, 1}`.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
        assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
        assert isinstance(camera, Cameras), f"{type(camera)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert isinstance(
            uv_rasterization_data, dict
        ), f"{type(uv_rasterization_data)=}"
        assert vertices.ndim == 2, f"{vertices.shape=}"
        assert vertices.shape[1] == 3, f"{vertices.shape=}"
        assert faces.ndim == 2, f"{faces.shape=}"
        assert faces.shape[1] == 3, f"{faces.shape=}"
        assert faces.dtype == torch.long, f"{faces.dtype=}"
        assert len(camera) == 1, f"{len(camera)=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"
        assert "uv_mask" in uv_rasterization_data, f"{uv_rasterization_data.keys()=}"
        assert "rast_out" in uv_rasterization_data, f"{uv_rasterization_data.keys()=}"
        assert (
            "raster_face_indices" in uv_rasterization_data
        ), f"{uv_rasterization_data.keys()=}"
        assert (
            "camera_attr_vertex_uv" in uv_rasterization_data
        ), f"{uv_rasterization_data.keys()=}"
        assert isinstance(
            uv_rasterization_data["uv_mask"], torch.Tensor
        ), f"{type(uv_rasterization_data['uv_mask'])=}"
        assert (
            uv_rasterization_data["uv_mask"].dtype == torch.float32
        ), f"{uv_rasterization_data['uv_mask'].dtype=}"
        assert (
            uv_rasterization_data["uv_mask"].shape[0] == 1
        ), f"{uv_rasterization_data['uv_mask'].shape=}"
        assert (
            uv_rasterization_data["uv_mask"].shape[3] == 1
        ), f"{uv_rasterization_data['uv_mask'].shape=}"
        assert vertices.device == faces.device, (
            "Expected `vertices` and `faces` to share a device. "
            f"{vertices.device=} {faces.device=}."
        )
        assert vertices.device == uv_rasterization_data["uv_mask"].device, (
            "Expected `vertices` and `uv_rasterization_data['uv_mask']` to share a "
            f"device. {vertices.device=} {uv_rasterization_data['uv_mask'].device=}."
        )

    _validate_inputs()

    valid_texel_mask = uv_rasterization_data["uv_mask"]

    (
        valid_texel_indices,
        continuous_uv_coords,
    ) = _map_valid_texels_to_continuous_uv_coords(
        valid_texel_mask=valid_texel_mask,
    )
    (
        texel_face_indices,
        barycentric_coords,
    ) = _map_continuous_uv_coords_to_barycentric_coords(
        continuous_uv_coords=continuous_uv_coords,
        valid_texel_indices=valid_texel_indices,
        uv_rasterization_data=uv_rasterization_data,
    )
    (
        valid_texel_indices,
        texel_face_indices,
        barycentric_coords,
    ) = _filter_texels_by_face_facing(
        valid_texel_indices=valid_texel_indices,
        texel_face_indices=texel_face_indices,
        barycentric_coords=barycentric_coords,
        vertices=vertices,
        faces=faces,
        camera=camera,
    )
    world_coords = _map_barycentric_coords_to_3d_world_coords(
        barycentric_coords=barycentric_coords,
        texel_face_indices=texel_face_indices,
        vertices=vertices,
        faces=faces,
    )
    mesh_diagonal = _compute_mesh_diagonal(vertices=vertices)
    return _compute_texel_visibility_mask_from_world_coords(
        world_coords=world_coords,
        valid_texel_indices=valid_texel_indices,
        valid_texel_mask=valid_texel_mask,
        mesh_diagonal=mesh_diagonal,
        camera=camera,
        image_height=image_height,
        image_width=image_width,
    )


def _map_valid_texels_to_continuous_uv_coords(
    valid_texel_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map valid texel centers to continuous UV coordinates.

    Args:
        valid_texel_mask: Binary valid-texel mask `[1, T, T, 1]`.

    Returns:
        Tuple of:
            valid texel indices `[N, 2]` in `(y, x)` order,
            continuous UV texel-center coordinates `[N, 2]` in `(u, v)` order.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(valid_texel_mask, torch.Tensor), f"{type(valid_texel_mask)=}"
        assert valid_texel_mask.ndim == 4, f"{valid_texel_mask.shape=}"
        assert valid_texel_mask.shape[0] == 1, f"{valid_texel_mask.shape=}"
        assert valid_texel_mask.shape[3] == 1, f"{valid_texel_mask.shape=}"
        assert valid_texel_mask.dtype == torch.float32, f"{valid_texel_mask.dtype=}"

    _validate_inputs()

    texture_height = int(valid_texel_mask.shape[1])
    texture_width = int(valid_texel_mask.shape[2])
    valid_texel_indices = torch.nonzero(
        valid_texel_mask[0, :, :, 0] > 0.5,
        as_tuple=False,
    )
    assert valid_texel_indices.ndim == 2, f"{valid_texel_indices.shape=}"
    assert valid_texel_indices.shape[1] == 2, f"{valid_texel_indices.shape=}"
    if valid_texel_indices.shape[0] == 0:
        return (
            torch.zeros((0, 2), device=valid_texel_mask.device, dtype=torch.long),
            torch.zeros((0, 2), device=valid_texel_mask.device, dtype=torch.float32),
        )

    valid_texel_indices_float = valid_texel_indices.to(dtype=torch.float32)
    continuous_u = (valid_texel_indices_float[:, 1] + 0.5) / float(texture_width)
    continuous_v = (valid_texel_indices_float[:, 0] + 0.5) / float(texture_height)
    continuous_uv_coords = torch.stack(
        [continuous_u, continuous_v],
        dim=1,
    ).contiguous()
    return valid_texel_indices.contiguous(), continuous_uv_coords


def _map_continuous_uv_coords_to_barycentric_coords(
    continuous_uv_coords: torch.Tensor,
    valid_texel_indices: torch.Tensor,
    uv_rasterization_data: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map continuous UV coordinates to owning-face barycentric coordinates.

    Args:
        continuous_uv_coords: Continuous UV texel-center coordinates `[N, 2]`.
        valid_texel_indices: Valid texel indices `[N, 2]` in `(y, x)` order.
        uv_rasterization_data: Precomputed UV rasterization tensors.

    Returns:
        Tuple of:
            owning original face indices `[N]`,
            barycentric coordinates `[N, 3]`.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(
            continuous_uv_coords, torch.Tensor
        ), f"{type(continuous_uv_coords)=}"
        assert isinstance(
            valid_texel_indices, torch.Tensor
        ), f"{type(valid_texel_indices)=}"
        assert isinstance(
            uv_rasterization_data, dict
        ), f"{type(uv_rasterization_data)=}"
        assert continuous_uv_coords.ndim == 2, f"{continuous_uv_coords.shape=}"
        assert continuous_uv_coords.shape[1] == 2, f"{continuous_uv_coords.shape=}"
        assert (
            continuous_uv_coords.dtype == torch.float32
        ), f"{continuous_uv_coords.dtype=}"
        assert valid_texel_indices.ndim == 2, f"{valid_texel_indices.shape=}"
        assert valid_texel_indices.shape[1] == 2, f"{valid_texel_indices.shape=}"
        assert valid_texel_indices.dtype == torch.long, f"{valid_texel_indices.dtype=}"
        assert (
            continuous_uv_coords.shape[0] == valid_texel_indices.shape[0]
        ), f"{continuous_uv_coords.shape=} {valid_texel_indices.shape=}"
        assert "rast_out" in uv_rasterization_data, f"{uv_rasterization_data.keys()=}"
        assert (
            "raster_face_indices" in uv_rasterization_data
        ), f"{uv_rasterization_data.keys()=}"
        assert (
            "camera_attr_vertex_uv" in uv_rasterization_data
        ), f"{uv_rasterization_data.keys()=}"
        assert isinstance(
            uv_rasterization_data["rast_out"], torch.Tensor
        ), f"{type(uv_rasterization_data['rast_out'])=}"
        assert isinstance(
            uv_rasterization_data["raster_face_indices"], torch.Tensor
        ), f"{type(uv_rasterization_data['raster_face_indices'])=}"
        assert isinstance(
            uv_rasterization_data["camera_attr_vertex_uv"], torch.Tensor
        ), f"{type(uv_rasterization_data['camera_attr_vertex_uv'])=}"
        assert (
            uv_rasterization_data["rast_out"].dtype == torch.float32
        ), f"{uv_rasterization_data['rast_out'].dtype=}"
        assert (
            uv_rasterization_data["raster_face_indices"].dtype == torch.long
        ), f"{uv_rasterization_data['raster_face_indices'].dtype=}"
        assert (
            uv_rasterization_data["camera_attr_vertex_uv"].dtype == torch.float32
        ), f"{uv_rasterization_data['camera_attr_vertex_uv'].dtype=}"
        assert (
            uv_rasterization_data["rast_out"].shape[0] == 1
        ), f"{uv_rasterization_data['rast_out'].shape=}"
        assert (
            uv_rasterization_data["rast_out"].shape[3] == 4
        ), f"{uv_rasterization_data['rast_out'].shape=}"
        assert (
            uv_rasterization_data["camera_attr_vertex_uv"].ndim == 2
        ), f"{uv_rasterization_data['camera_attr_vertex_uv'].shape=}"
        assert (
            uv_rasterization_data["camera_attr_vertex_uv"].shape[1] == 2
        ), f"{uv_rasterization_data['camera_attr_vertex_uv'].shape=}"
        assert (
            uv_rasterization_data["camera_attr_vertex_uv"].shape[0] % 3 == 0
        ), f"{uv_rasterization_data['camera_attr_vertex_uv'].shape=}"
        assert (
            continuous_uv_coords.device == uv_rasterization_data["rast_out"].device
        ), (
            "Expected `continuous_uv_coords` and `uv_rasterization_data['rast_out']` "
            f"to share a device. {continuous_uv_coords.device=} "
            f"{uv_rasterization_data['rast_out'].device=}."
        )
        assert (
            continuous_uv_coords.device
            == uv_rasterization_data["raster_face_indices"].device
        ), (
            "Expected `continuous_uv_coords` and "
            "`uv_rasterization_data['raster_face_indices']` to share a device. "
            f"{continuous_uv_coords.device=} "
            f"{uv_rasterization_data['raster_face_indices'].device=}."
        )
        assert (
            continuous_uv_coords.device
            == uv_rasterization_data["camera_attr_vertex_uv"].device
        ), (
            "Expected `continuous_uv_coords` and "
            "`uv_rasterization_data['camera_attr_vertex_uv']` to share a device. "
            f"{continuous_uv_coords.device=} "
            f"{uv_rasterization_data['camera_attr_vertex_uv'].device=}."
        )

    _validate_inputs()

    if continuous_uv_coords.shape[0] == 0:
        return (
            torch.zeros((0,), device=continuous_uv_coords.device, dtype=torch.long),
            torch.zeros(
                (0, 3), device=continuous_uv_coords.device, dtype=torch.float32
            ),
        )

    rast_out = uv_rasterization_data["rast_out"]
    raster_face_indices = uv_rasterization_data["raster_face_indices"]
    camera_attr_vertex_uv = uv_rasterization_data["camera_attr_vertex_uv"]

    uv_triangle_indices_float = rast_out[
        0,
        valid_texel_indices[:, 0],
        valid_texel_indices[:, 1],
        3,
    ]
    assert (
        uv_triangle_indices_float.dtype == torch.float32
    ), f"{uv_triangle_indices_float.dtype=}"
    uv_triangle_indices = uv_triangle_indices_float.to(dtype=torch.long) - 1
    assert torch.all(uv_triangle_indices >= 0), (
        "Expected all valid texels to map to an owning UV triangle. "
        f"{uv_triangle_indices.min()=}"
    )
    texel_face_indices = raster_face_indices[uv_triangle_indices]
    face_vertex_uv = camera_attr_vertex_uv.reshape(-1, 3, 2)[texel_face_indices]
    wrapped_continuous_uv_coords = _wrap_continuous_uv_coords_for_faces(
        continuous_uv_coords=continuous_uv_coords,
        face_vertex_uv=face_vertex_uv,
    )
    barycentric_coords = _compute_barycentric_coords_in_uv_faces(
        continuous_uv_coords=wrapped_continuous_uv_coords,
        face_vertex_uv=face_vertex_uv,
    )
    return texel_face_indices.contiguous(), barycentric_coords


def _filter_texels_by_face_facing(
    valid_texel_indices: torch.Tensor,
    texel_face_indices: torch.Tensor,
    barycentric_coords: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    camera: Cameras,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter texels whose owning mesh face is back-facing in the current view.

    Args:
        valid_texel_indices: Valid texel indices `[N, 2]` in `(y, x)` order.
        texel_face_indices: Owning original face indices `[N]`.
        barycentric_coords: Barycentric coordinates `[N, 3]`.
        vertices: Mesh vertices `[V, 3]`.
        faces: Mesh faces `[F, 3]`.
        camera: One camera instance.

    Returns:
        Tuple of filtered valid texel indices `[M, 2]`, filtered owning face
        indices `[M]`, and filtered barycentric coordinates `[M, 3]`.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(
            valid_texel_indices, torch.Tensor
        ), f"Expected `valid_texel_indices` to be a tensor. {type(valid_texel_indices)=}."
        assert isinstance(
            texel_face_indices, torch.Tensor
        ), f"Expected `texel_face_indices` to be a tensor. {type(texel_face_indices)=}."
        assert isinstance(
            barycentric_coords, torch.Tensor
        ), f"Expected `barycentric_coords` to be a tensor. {type(barycentric_coords)=}."
        assert isinstance(
            vertices, torch.Tensor
        ), f"Expected `vertices` to be a tensor. {type(vertices)=}."
        assert isinstance(
            faces, torch.Tensor
        ), f"Expected `faces` to be a tensor. {type(faces)=}."
        assert isinstance(
            camera, Cameras
        ), f"Expected `camera` to be a `Cameras` instance. {type(camera)=}."
        assert valid_texel_indices.ndim == 2, (
            "Expected `valid_texel_indices` to have shape `[N, 2]`. "
            f"{valid_texel_indices.shape=}."
        )
        assert valid_texel_indices.shape[1] == 2, (
            "Expected `valid_texel_indices` to have shape `[N, 2]`. "
            f"{valid_texel_indices.shape=}."
        )
        assert valid_texel_indices.dtype == torch.long, (
            "Expected `valid_texel_indices` to have dtype `torch.long`. "
            f"{valid_texel_indices.dtype=}."
        )
        assert texel_face_indices.ndim == 1, (
            "Expected `texel_face_indices` to have shape `[N]`. "
            f"{texel_face_indices.shape=}."
        )
        assert texel_face_indices.dtype == torch.long, (
            "Expected `texel_face_indices` to have dtype `torch.long`. "
            f"{texel_face_indices.dtype=}."
        )
        assert barycentric_coords.ndim == 2, (
            "Expected `barycentric_coords` to have shape `[N, 3]`. "
            f"{barycentric_coords.shape=}."
        )
        assert barycentric_coords.shape[1] == 3, (
            "Expected `barycentric_coords` to have shape `[N, 3]`. "
            f"{barycentric_coords.shape=}."
        )
        assert barycentric_coords.dtype == torch.float32, (
            "Expected `barycentric_coords` to have dtype `torch.float32`. "
            f"{barycentric_coords.dtype=}."
        )
        assert vertices.ndim == 2, (
            "Expected `vertices` to have shape `[V, 3]`. " f"{vertices.shape=}."
        )
        assert vertices.shape[1] == 3, (
            "Expected `vertices` to have shape `[V, 3]`. " f"{vertices.shape=}."
        )
        assert vertices.dtype == torch.float32, (
            "Expected `vertices` to have dtype `torch.float32`. " f"{vertices.dtype=}."
        )
        assert faces.ndim == 2, (
            "Expected `faces` to have shape `[F, 3]`. " f"{faces.shape=}."
        )
        assert faces.shape[1] == 3, (
            "Expected `faces` to have shape `[F, 3]`. " f"{faces.shape=}."
        )
        assert faces.dtype == torch.long, (
            "Expected `faces` to have dtype `torch.long`. " f"{faces.dtype=}."
        )
        assert len(camera) == 1, (
            "Expected `camera` to contain exactly one view. " f"{len(camera)=}."
        )
        assert valid_texel_indices.shape[0] == texel_face_indices.shape[0], (
            "Expected `valid_texel_indices` and `texel_face_indices` to agree on "
            f"texel count. {valid_texel_indices.shape=} {texel_face_indices.shape=}."
        )
        assert barycentric_coords.shape[0] == texel_face_indices.shape[0], (
            "Expected `barycentric_coords` and `texel_face_indices` to agree on "
            f"texel count. {barycentric_coords.shape=} {texel_face_indices.shape=}."
        )
        assert valid_texel_indices.device == texel_face_indices.device, (
            "Expected `valid_texel_indices` and `texel_face_indices` to share a "
            f"device. {valid_texel_indices.device=} {texel_face_indices.device=}."
        )
        assert valid_texel_indices.device == barycentric_coords.device, (
            "Expected `valid_texel_indices` and `barycentric_coords` to share a "
            f"device. {valid_texel_indices.device=} {barycentric_coords.device=}."
        )
        assert valid_texel_indices.device == vertices.device, (
            "Expected `valid_texel_indices` and `vertices` to share a device. "
            f"{valid_texel_indices.device=} {vertices.device=}."
        )
        assert valid_texel_indices.device == faces.device, (
            "Expected `valid_texel_indices` and `faces` to share a device. "
            f"{valid_texel_indices.device=} {faces.device=}."
        )

    _validate_inputs()

    if texel_face_indices.shape[0] == 0:
        return valid_texel_indices, texel_face_indices, barycentric_coords

    face_front_facing_mask = (
        _compute_f_normals_weights(
            mesh=Mesh(vertices=vertices, faces=faces),
            camera=camera,
            weights_cfg={"weights": "normals"},
        )
        > 0.0
    )
    assert face_front_facing_mask.ndim == 1, (
        "Expected `face_front_facing_mask` to have shape `[F]`. "
        f"{face_front_facing_mask.shape=}."
    )
    assert face_front_facing_mask.shape[0] == faces.shape[0], (
        "Expected `face_front_facing_mask` to align with `faces`. "
        f"{face_front_facing_mask.shape=} {faces.shape=}."
    )
    assert face_front_facing_mask.dtype == torch.bool, (
        "Expected `face_front_facing_mask` to have dtype `torch.bool`. "
        f"{face_front_facing_mask.dtype=}."
    )

    texel_front_facing_mask = face_front_facing_mask[texel_face_indices]
    assert texel_front_facing_mask.ndim == 1, (
        "Expected `texel_front_facing_mask` to have shape `[N]`. "
        f"{texel_front_facing_mask.shape=}."
    )
    assert texel_front_facing_mask.shape[0] == texel_face_indices.shape[0], (
        "Expected `texel_front_facing_mask` to align with `texel_face_indices`. "
        f"{texel_front_facing_mask.shape=} {texel_face_indices.shape=}."
    )

    return (
        valid_texel_indices[texel_front_facing_mask].contiguous(),
        texel_face_indices[texel_front_facing_mask].contiguous(),
        barycentric_coords[texel_front_facing_mask].contiguous(),
    )


def _map_barycentric_coords_to_3d_world_coords(
    barycentric_coords: torch.Tensor,
    texel_face_indices: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Map barycentric texel coordinates to world-space mesh points.

    Args:
        barycentric_coords: Barycentric coordinates `[N, 3]`.
        texel_face_indices: Owning original face indices `[N]`.
        vertices: Mesh vertices `[V, 3]`.
        faces: Mesh faces `[F, 3]`.

    Returns:
        World-space texel-center points `[N, 3]`.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(
            barycentric_coords, torch.Tensor
        ), f"{type(barycentric_coords)=}"
        assert isinstance(
            texel_face_indices, torch.Tensor
        ), f"{type(texel_face_indices)=}"
        assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
        assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
        assert barycentric_coords.ndim == 2, f"{barycentric_coords.shape=}"
        assert barycentric_coords.shape[1] == 3, f"{barycentric_coords.shape=}"
        assert barycentric_coords.dtype == torch.float32, f"{barycentric_coords.dtype=}"
        assert texel_face_indices.ndim == 1, f"{texel_face_indices.shape=}"
        assert texel_face_indices.dtype == torch.long, f"{texel_face_indices.dtype=}"
        assert (
            barycentric_coords.shape[0] == texel_face_indices.shape[0]
        ), f"{barycentric_coords.shape=} {texel_face_indices.shape=}"
        assert vertices.ndim == 2, f"{vertices.shape=}"
        assert vertices.shape[1] == 3, f"{vertices.shape=}"
        assert vertices.dtype == torch.float32, f"{vertices.dtype=}"
        assert faces.ndim == 2, f"{faces.shape=}"
        assert faces.shape[1] == 3, f"{faces.shape=}"
        assert faces.dtype == torch.long, f"{faces.dtype=}"
        assert barycentric_coords.device == vertices.device, (
            "Expected `barycentric_coords` and `vertices` to share a device. "
            f"{barycentric_coords.device=} {vertices.device=}."
        )
        assert barycentric_coords.device == faces.device, (
            "Expected `barycentric_coords` and `faces` to share a device. "
            f"{barycentric_coords.device=} {faces.device=}."
        )

    _validate_inputs()

    if barycentric_coords.shape[0] == 0:
        return torch.zeros((0, 3), device=vertices.device, dtype=torch.float32)

    face_vertices = vertices[faces[texel_face_indices]]
    world_coords = (
        barycentric_coords[:, 0:1] * face_vertices[:, 0, :]
        + barycentric_coords[:, 1:2] * face_vertices[:, 1, :]
        + barycentric_coords[:, 2:3] * face_vertices[:, 2, :]
    )
    return world_coords.contiguous()


def _compute_texel_visibility_mask_from_world_coords(
    world_coords: torch.Tensor,
    valid_texel_indices: torch.Tensor,
    valid_texel_mask: torch.Tensor,
    mesh_diagonal: float,
    camera: Cameras,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Compute texel visibility by keeping the front depth-prefix per pixel.

    Args:
        world_coords: World-space texel-center points `[N, 3]`.
        valid_texel_indices: Valid texel indices `[N, 2]` in `(y, x)` order.
        valid_texel_mask: Binary valid-texel mask `[1, T, T, 1]`.
        mesh_diagonal: Full-mesh diagonal length.
        camera: One camera instance.
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        Float tensor `[1, T, T, 1]` with values in `{0, 1}`.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(world_coords, torch.Tensor), f"{type(world_coords)=}"
        assert isinstance(
            valid_texel_indices, torch.Tensor
        ), f"{type(valid_texel_indices)=}"
        assert isinstance(valid_texel_mask, torch.Tensor), f"{type(valid_texel_mask)=}"
        assert isinstance(mesh_diagonal, float), f"{type(mesh_diagonal)=}"
        assert isinstance(camera, Cameras), f"{type(camera)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert world_coords.ndim == 2, f"{world_coords.shape=}"
        assert world_coords.shape[1] == 3, f"{world_coords.shape=}"
        assert world_coords.dtype == torch.float32, f"{world_coords.dtype=}"
        assert valid_texel_indices.ndim == 2, f"{valid_texel_indices.shape=}"
        assert valid_texel_indices.shape[1] == 2, f"{valid_texel_indices.shape=}"
        assert valid_texel_indices.dtype == torch.long, f"{valid_texel_indices.dtype=}"
        assert (
            world_coords.shape[0] == valid_texel_indices.shape[0]
        ), f"{world_coords.shape=} {valid_texel_indices.shape=}"
        assert valid_texel_mask.ndim == 4, f"{valid_texel_mask.shape=}"
        assert valid_texel_mask.shape[0] == 1, f"{valid_texel_mask.shape=}"
        assert valid_texel_mask.shape[3] == 1, f"{valid_texel_mask.shape=}"
        assert valid_texel_mask.dtype == torch.float32, f"{valid_texel_mask.dtype=}"
        assert mesh_diagonal > 0.0, f"{mesh_diagonal=}"
        assert len(camera) == 1, f"{len(camera)=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"
        assert world_coords.device == valid_texel_mask.device, (
            "Expected `world_coords` and `valid_texel_mask` to share a device. "
            f"{world_coords.device=} {valid_texel_mask.device=}."
        )

    _validate_inputs()

    visibility_mask = torch.zeros_like(valid_texel_mask)
    if world_coords.shape[0] == 0:
        return visibility_mask

    camera_single = camera[0].to(
        device=world_coords.device,
        convention="opencv",
    )
    texel_camera_coords = world_to_camera_transform(
        points=world_coords,
        extrinsics=camera_single.extrinsics,
        inplace=False,
    )
    assert texel_camera_coords.dtype == torch.float32, f"{texel_camera_coords.dtype=}"
    projected_texel_points = project_3d_to_2d(
        points=texel_camera_coords,
        intrinsics=camera_single.intrinsics,
        inplace=False,
    )
    assert (
        projected_texel_points.dtype == torch.float32
    ), f"{projected_texel_points.dtype=}"
    projected_depth = texel_camera_coords[:, 2]
    projected_x = projected_texel_points[:, 0]
    projected_y = projected_texel_points[:, 1]
    projection_valid_mask = (
        (projected_depth > 0.0)
        & (projected_x >= 0.0)
        & (projected_x < float(image_width))
        & (projected_y >= 0.0)
        & (projected_y < float(image_height))
    )
    if not bool(projection_valid_mask.any()):
        return visibility_mask

    visible_projected_depth = projected_depth[projection_valid_mask]
    visible_texel_indices = valid_texel_indices[projection_valid_mask]
    visible_projected_x = projected_x[projection_valid_mask]
    visible_projected_y = projected_y[projection_valid_mask]
    visible_pixel_x = torch.floor(visible_projected_x).to(dtype=torch.long)
    visible_pixel_y = torch.floor(visible_projected_y).to(dtype=torch.long)
    visible_linear_pixel_indices = visible_pixel_y * image_width + visible_pixel_x
    visible_selection_mask = _select_visible_depth_clusters_per_camera_pixel(
        linear_pixel_indices=visible_linear_pixel_indices,
        depth=visible_projected_depth,
        mesh_diagonal=mesh_diagonal,
    )
    visibility_mask[
        0,
        visible_texel_indices[visible_selection_mask, 0],
        visible_texel_indices[visible_selection_mask, 1],
        0,
    ] = 1.0
    return visibility_mask.contiguous()


# -----------------------------------------------------------------------------
# Chunk 2: Internal Helpers
# -----------------------------------------------------------------------------


def _wrap_continuous_uv_coords_for_faces(
    continuous_uv_coords: torch.Tensor,
    face_vertex_uv: torch.Tensor,
) -> torch.Tensor:
    """Wrap texel-center UV coordinates into the seam-safe face-local chart.

    Args:
        continuous_uv_coords: Continuous UV texel-center coordinates `[N, 2]`.
        face_vertex_uv: Seam-safe face UV triangles `[N, 3, 2]`.

    Returns:
        Wrapped continuous UV coordinates `[N, 2]`.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(
            continuous_uv_coords, torch.Tensor
        ), f"{type(continuous_uv_coords)=}"
        assert isinstance(face_vertex_uv, torch.Tensor), f"{type(face_vertex_uv)=}"
        assert continuous_uv_coords.ndim == 2, f"{continuous_uv_coords.shape=}"
        assert continuous_uv_coords.shape[1] == 2, f"{continuous_uv_coords.shape=}"
        assert (
            continuous_uv_coords.dtype == torch.float32
        ), f"{continuous_uv_coords.dtype=}"
        assert face_vertex_uv.ndim == 3, f"{face_vertex_uv.shape=}"
        assert face_vertex_uv.shape[1] == 3, f"{face_vertex_uv.shape=}"
        assert face_vertex_uv.shape[2] == 2, f"{face_vertex_uv.shape=}"
        assert face_vertex_uv.dtype == torch.float32, f"{face_vertex_uv.dtype=}"
        assert (
            continuous_uv_coords.shape[0] == face_vertex_uv.shape[0]
        ), f"{continuous_uv_coords.shape=} {face_vertex_uv.shape=}"

    _validate_inputs()

    if continuous_uv_coords.shape[0] == 0:
        return continuous_uv_coords

    wrapped_continuous_uv_coords = continuous_uv_coords.clone()
    face_u = face_vertex_uv[:, :, 0]
    seam_face_mask = face_u.max(dim=1).values > 1.0
    if bool(seam_face_mask.any()):
        wrapped_continuous_uv_coords[seam_face_mask, 0] = torch.where(
            wrapped_continuous_uv_coords[seam_face_mask, 0] < 0.5,
            wrapped_continuous_uv_coords[seam_face_mask, 0] + 1.0,
            wrapped_continuous_uv_coords[seam_face_mask, 0],
        )
    return wrapped_continuous_uv_coords.contiguous()


def _compute_barycentric_coords_in_uv_faces(
    continuous_uv_coords: torch.Tensor,
    face_vertex_uv: torch.Tensor,
) -> torch.Tensor:
    """Compute barycentric coordinates of points inside UV triangles.

    Args:
        continuous_uv_coords: Continuous UV texel-center coordinates `[N, 2]`.
        face_vertex_uv: Seam-safe face UV triangles `[N, 3, 2]`.

    Returns:
        Barycentric coordinates `[N, 3]`.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(
            continuous_uv_coords, torch.Tensor
        ), f"{type(continuous_uv_coords)=}"
        assert isinstance(face_vertex_uv, torch.Tensor), f"{type(face_vertex_uv)=}"
        assert continuous_uv_coords.ndim == 2, f"{continuous_uv_coords.shape=}"
        assert continuous_uv_coords.shape[1] == 2, f"{continuous_uv_coords.shape=}"
        assert (
            continuous_uv_coords.dtype == torch.float32
        ), f"{continuous_uv_coords.dtype=}"
        assert face_vertex_uv.ndim == 3, f"{face_vertex_uv.shape=}"
        assert face_vertex_uv.shape[1] == 3, f"{face_vertex_uv.shape=}"
        assert face_vertex_uv.shape[2] == 2, f"{face_vertex_uv.shape=}"
        assert face_vertex_uv.dtype == torch.float32, f"{face_vertex_uv.dtype=}"
        assert (
            continuous_uv_coords.shape[0] == face_vertex_uv.shape[0]
        ), f"{continuous_uv_coords.shape=} {face_vertex_uv.shape=}"

    _validate_inputs()

    if continuous_uv_coords.shape[0] == 0:
        return torch.zeros(
            (0, 3), device=continuous_uv_coords.device, dtype=torch.float32
        )

    vertex0 = face_vertex_uv[:, 0, :]
    vertex1 = face_vertex_uv[:, 1, :]
    vertex2 = face_vertex_uv[:, 2, :]
    edge01 = vertex1 - vertex0
    edge02 = vertex2 - vertex0
    point_offset = continuous_uv_coords - vertex0

    determinant = edge01[:, 0] * edge02[:, 1] - edge01[:, 1] * edge02[:, 0]
    assert torch.all(torch.abs(determinant) > 1e-12), (
        "Expected all owning UV triangles to be non-degenerate. "
        f"{determinant.min()=}"
    )
    barycentric1 = (
        point_offset[:, 0] * edge02[:, 1] - point_offset[:, 1] * edge02[:, 0]
    ) / determinant
    barycentric2 = (
        edge01[:, 0] * point_offset[:, 1] - edge01[:, 1] * point_offset[:, 0]
    ) / determinant
    barycentric0 = 1.0 - barycentric1 - barycentric2
    barycentric_coords = torch.stack(
        [barycentric0, barycentric1, barycentric2],
        dim=1,
    )
    return barycentric_coords.contiguous()


def _compute_mesh_diagonal(vertices: torch.Tensor) -> float:
    """Compute the full-mesh diagonal length.

    Args:
        vertices: Mesh vertices `[V, 3]`.

    Returns:
        Mesh diagonal length as a Python float.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
        assert vertices.ndim == 2, f"{vertices.shape=}"
        assert vertices.shape[1] == 3, f"{vertices.shape=}"
        assert vertices.shape[0] > 0, f"{vertices.shape=}"

    _validate_inputs()

    min_vertices = vertices.min(dim=0).values
    max_vertices = vertices.max(dim=0).values
    mesh_diagonal = float(torch.linalg.norm(max_vertices - min_vertices).item())
    assert mesh_diagonal > 0.0, f"{mesh_diagonal=}"
    return mesh_diagonal


def _select_visible_depth_clusters_per_camera_pixel(
    linear_pixel_indices: torch.Tensor,
    depth: torch.Tensor,
    mesh_diagonal: float,
) -> torch.Tensor:
    """Keep only the first front depth cluster in each pixel stack.

    Args:
        linear_pixel_indices: Linearized camera-pixel ids `[N]`.
        depth: Camera-space texel depths `[N]`.
        mesh_diagonal: Mesh diagonal length used to normalize depth gaps.

    Returns:
        Boolean selection mask `[N]` over the input texels.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(
            linear_pixel_indices, torch.Tensor
        ), f"{type(linear_pixel_indices)=}"
        assert isinstance(depth, torch.Tensor), f"{type(depth)=}"
        assert isinstance(mesh_diagonal, float), f"{type(mesh_diagonal)=}"
        assert linear_pixel_indices.ndim == 1, f"{linear_pixel_indices.shape=}"
        assert depth.ndim == 1, f"{depth.shape=}"
        assert (
            linear_pixel_indices.shape == depth.shape
        ), f"{linear_pixel_indices.shape=} {depth.shape=}"
        assert (
            linear_pixel_indices.dtype == torch.long
        ), f"{linear_pixel_indices.dtype=}"
        assert depth.dtype == torch.float32, f"{depth.dtype=}"
        assert mesh_diagonal > 0.0, f"{mesh_diagonal=}"

    _validate_inputs()

    if linear_pixel_indices.numel() == 0:
        return torch.zeros_like(linear_pixel_indices, dtype=torch.bool)

    (
        sorted_indices,
        sorted_linear_pixel_indices,
        sorted_depth,
        segment_start_mask,
    ) = _sort_depth_stacks_per_camera_pixel(
        linear_pixel_indices=linear_pixel_indices,
        depth=depth,
    )
    depth_gap_threshold_relative = _compute_front_depth_gap_threshold_relative(
        sorted_depth=sorted_depth,
        segment_start_mask=segment_start_mask,
        mesh_diagonal=mesh_diagonal,
    )
    depth_gap_from_previous = torch.zeros_like(sorted_depth)
    depth_gap_from_previous[1:] = (sorted_depth[1:] - sorted_depth[:-1]) / mesh_diagonal
    gap_break_mask = (~segment_start_mask) & (
        depth_gap_from_previous > depth_gap_threshold_relative
    )
    cluster_start_mask = segment_start_mask | gap_break_mask
    cluster_start_indices = torch.nonzero(cluster_start_mask, as_tuple=False).flatten()
    assert cluster_start_indices.ndim == 1, (
        "Expected `cluster_start_indices` to be rank-1. "
        f"{cluster_start_indices.shape=}"
    )
    assert cluster_start_indices.shape[0] > 0, (
        "Expected at least one cluster start in the sorted stacks. "
        f"{cluster_start_indices.shape=}"
    )
    cluster_end_indices = torch.empty_like(cluster_start_indices)
    cluster_end_indices[:-1] = cluster_start_indices[1:]
    cluster_end_indices[-1] = int(sorted_depth.shape[0])
    pixel_start_indices = torch.nonzero(segment_start_mask, as_tuple=False).flatten()
    assert pixel_start_indices.ndim == 1, (
        "Expected `pixel_start_indices` to be rank-1. " f"{pixel_start_indices.shape=}"
    )
    assert pixel_start_indices.shape[0] > 0, (
        "Expected at least one pixel stack start. " f"{pixel_start_indices.shape=}"
    )
    pixel_end_indices = torch.empty_like(pixel_start_indices)
    pixel_end_indices[:-1] = pixel_start_indices[1:]
    pixel_end_indices[-1] = int(sorted_depth.shape[0])
    pixel_stack_sizes = pixel_end_indices - pixel_start_indices
    assert torch.all(pixel_stack_sizes > 0), (
        "Expected every pixel stack to have positive size. " f"{pixel_stack_sizes=}"
    )

    cluster_start_pixels = sorted_linear_pixel_indices[cluster_start_indices]
    pixel_start_positions_in_cluster_starts = torch.nonzero(
        segment_start_mask[cluster_start_indices],
        as_tuple=False,
    ).flatten()
    assert pixel_start_positions_in_cluster_starts.shape == pixel_start_indices.shape, (
        "Expected one cluster-start position per pixel stack. "
        f"{pixel_start_positions_in_cluster_starts.shape=} "
        f"{pixel_start_indices.shape=}"
    )
    next_cluster_positions = pixel_start_positions_in_cluster_starts + 1
    next_cluster_is_same_pixel = next_cluster_positions < cluster_start_indices.shape[0]
    next_cluster_is_same_pixel = next_cluster_is_same_pixel & (
        cluster_start_pixels[
            torch.clamp(
                next_cluster_positions,
                max=cluster_start_indices.shape[0] - 1,
            )
        ]
        == cluster_start_pixels[pixel_start_positions_in_cluster_starts]
    )
    first_cluster_end_indices = torch.where(
        next_cluster_is_same_pixel,
        cluster_start_indices[
            torch.clamp(
                next_cluster_positions,
                max=cluster_start_indices.shape[0] - 1,
            )
        ],
        pixel_end_indices,
    )
    repeated_first_cluster_end_indices = torch.repeat_interleave(
        first_cluster_end_indices,
        pixel_stack_sizes,
    )
    assert repeated_first_cluster_end_indices.shape == sorted_depth.shape, (
        "Expected repeated first-cluster end indices to align with the sorted "
        "texels. "
        f"{repeated_first_cluster_end_indices.shape=} {sorted_depth.shape=}"
    )
    sorted_positions = torch.arange(
        sorted_depth.shape[0],
        device=sorted_depth.device,
        dtype=torch.long,
    )
    sorted_selection_mask = sorted_positions < repeated_first_cluster_end_indices
    selection_mask = torch.zeros_like(sorted_selection_mask)
    selection_mask[sorted_indices] = sorted_selection_mask
    return selection_mask.contiguous()


def _sort_depth_stacks_per_camera_pixel(
    linear_pixel_indices: torch.Tensor,
    depth: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort projected texels into per-pixel depth stacks.

    Args:
        linear_pixel_indices: Linearized camera-pixel ids `[N]`.
        depth: Camera-space texel depths `[N]`.

    Returns:
        Tuple of:
            sorted input indices `[N]`,
            sorted linear pixel ids `[N]`,
            sorted depths `[N]`,
            segment-start mask `[N]`.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(
            linear_pixel_indices, torch.Tensor
        ), f"{type(linear_pixel_indices)=}"
        assert isinstance(depth, torch.Tensor), f"{type(depth)=}"
        assert linear_pixel_indices.ndim == 1, f"{linear_pixel_indices.shape=}"
        assert depth.ndim == 1, f"{depth.shape=}"
        assert (
            linear_pixel_indices.shape == depth.shape
        ), f"{linear_pixel_indices.shape=} {depth.shape=}"
        assert (
            linear_pixel_indices.dtype == torch.long
        ), f"{linear_pixel_indices.dtype=}"
        assert depth.dtype == torch.float32, f"{depth.dtype=}"

    _validate_inputs()

    if linear_pixel_indices.numel() == 0:
        empty_long = torch.zeros(
            (0,), device=linear_pixel_indices.device, dtype=torch.long
        )
        empty_float = torch.zeros((0,), device=depth.device, dtype=torch.float32)
        empty_bool = torch.zeros(
            (0,), device=linear_pixel_indices.device, dtype=torch.bool
        )
        return empty_long, empty_long, empty_float, empty_bool

    depth_order = torch.argsort(depth, stable=True)
    pixel_sorted_linear_pixel_indices = linear_pixel_indices[depth_order]
    pixel_order = torch.argsort(pixel_sorted_linear_pixel_indices, stable=True)
    sorted_indices = depth_order[pixel_order]
    sorted_linear_pixel_indices = linear_pixel_indices[sorted_indices]
    sorted_depth = depth[sorted_indices]
    segment_start_mask = torch.ones_like(sorted_linear_pixel_indices, dtype=torch.bool)
    segment_start_mask[1:] = (
        sorted_linear_pixel_indices[1:] != sorted_linear_pixel_indices[:-1]
    )
    return (
        sorted_indices.contiguous(),
        sorted_linear_pixel_indices.contiguous(),
        sorted_depth.contiguous(),
        segment_start_mask.contiguous(),
    )


def _compute_front_depth_gap_threshold_relative(
    sorted_depth: torch.Tensor,
    segment_start_mask: torch.Tensor,
    mesh_diagonal: float,
) -> float:
    """Derive the front-depth stopping threshold from the gap distribution.

    Args:
        sorted_depth: Per-pixel depth stacks sorted from near to far `[N]`.
        segment_start_mask: Boolean mask `[N]` marking stack starts.
        mesh_diagonal: Mesh diagonal length used to normalize depth gaps.

    Returns:
        Relative consecutive-gap threshold as a Python float.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(sorted_depth, torch.Tensor), f"{type(sorted_depth)=}"
        assert isinstance(
            segment_start_mask, torch.Tensor
        ), f"{type(segment_start_mask)=}"
        assert isinstance(mesh_diagonal, float), f"{type(mesh_diagonal)=}"
        assert sorted_depth.ndim == 1, f"{sorted_depth.shape=}"
        assert segment_start_mask.ndim == 1, f"{segment_start_mask.shape=}"
        assert (
            sorted_depth.shape == segment_start_mask.shape
        ), f"{sorted_depth.shape=} {segment_start_mask.shape=}"
        assert sorted_depth.dtype == torch.float32, f"{sorted_depth.dtype=}"
        assert segment_start_mask.dtype == torch.bool, f"{segment_start_mask.dtype=}"
        assert mesh_diagonal > 0.0, f"{mesh_diagonal=}"

    _validate_inputs()

    if sorted_depth.numel() <= 1:
        return 0.0

    relative_depth_gap_from_previous = torch.zeros_like(sorted_depth)
    relative_depth_gap_from_previous[1:] = (
        sorted_depth[1:] - sorted_depth[:-1]
    ) / mesh_diagonal
    positive_relative_depth_gaps = relative_depth_gap_from_previous[
        (~segment_start_mask) & (relative_depth_gap_from_previous > 0.0)
    ]
    if positive_relative_depth_gaps.numel() == 0:
        return 0.0

    log_positive_relative_depth_gaps = torch.log10(positive_relative_depth_gaps)
    log_gap_median = torch.median(log_positive_relative_depth_gaps)
    log_gap_mad = torch.median(
        torch.abs(log_positive_relative_depth_gaps - log_gap_median)
    )
    log_gap_threshold = (
        log_gap_median + FRONT_DEPTH_GAP_LOG_MAD_MULTIPLIER * log_gap_mad
    )
    return float(10 ** float(log_gap_threshold.item()))
