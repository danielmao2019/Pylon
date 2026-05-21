from typing import Any, Optional

import torch

from data.structures.three_d.mesh.texture.mesh_texture import MeshTexture
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)


def validate_vertices(obj: Any) -> None:
    """Validate one mesh vertex tensor.

    Args:
        obj: Candidate vertex tensor with shape `[V, 3]`.

    Returns:
        None.
    """

    assert isinstance(obj, torch.Tensor), (
        "Expected `vertices` to be a `torch.Tensor`. " f"{type(obj)=}"
    )
    assert obj.ndim == 2, "Expected `vertices` to be rank 2. " f"{obj.shape=}"
    assert obj.shape[1] == 3, (
        "Expected `vertices` to have XYZ coordinates in the last dimension. "
        f"{obj.shape=}"
    )
    assert obj.shape[0] > 0, (
        "Expected `vertices` to contain at least one vertex. " f"{obj.shape=}"
    )
    assert obj.is_floating_point(), (
        "Expected `vertices` to use a floating dtype. " f"{obj.dtype=}"
    )
    assert torch.isfinite(obj).all(), (
        "Expected `vertices` to contain only finite values. "
        f"{obj.shape=} {obj.dtype=}"
    )


def validate_faces(obj: Any) -> None:
    """Validate one mesh face tensor.

    Args:
        obj: Candidate face tensor with shape `[F, 3]`.

    Returns:
        None.
    """

    assert isinstance(obj, torch.Tensor), (
        "Expected `faces` to be a `torch.Tensor`. " f"{type(obj)=}"
    )
    assert obj.ndim == 2, "Expected `faces` to be rank 2. " f"{obj.shape=}"
    assert obj.shape[1] == 3, (
        "Expected `faces` to contain triangular indices. " f"{obj.shape=}"
    )
    assert obj.shape[0] > 0, (
        "Expected `faces` to contain at least one face. " f"{obj.shape=}"
    )
    assert not obj.is_floating_point(), (
        "Expected `faces` to use an integer dtype. " f"{obj.dtype=}"
    )
    assert obj.dtype != torch.bool, (
        "Expected `faces` to use an integer index dtype, not bool. " f"{obj.dtype=}"
    )
    assert int(obj.min().item()) >= 0, (
        "Expected `faces` to contain only non-negative indices. "
        f"{int(obj.min().item())=}"
    )


def _validate_device_compatible(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    texture: Optional[MeshTexture],
) -> None:
    """Validate that the geometry and texture tensors live on one device.

    Args:
        vertices: Mesh vertex tensor `[V, 3]`.
        faces: Mesh face tensor `[F, 3]`.
        texture: Optional mesh texture whose tensors must share the vertices'
            device.

    Returns:
        None.
    """

    assert faces.device == vertices.device, (
        "Expected `faces` to live on the same device as `vertices`. "
        f"{faces.device=} {vertices.device=}"
    )
    if texture is not None:
        assert texture.device == vertices.device, (
            "Expected the mesh texture to live on the same device as `vertices`. "
            f"{texture.device=} {vertices.device=}"
        )


def validate_mesh_attributes(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    texture: Optional[MeshTexture] = None,
) -> None:
    """Validate the geometry and the texture<->geometry linkage for one mesh.

    The texture self-validates its own internal shapes in its constructor; this
    function validates the geometry and the linkage between texture and
    geometry.

    Args:
        vertices: Mesh vertex tensor `[V, 3]`.
        faces: Mesh face tensor `[F, 3]`.
        texture: Optional mesh texture (`MeshTextureVertexColor` or
            `MeshTextureUVTextureMap`).

    Returns:
        None.
    """

    validate_vertices(obj=vertices)
    validate_faces(obj=faces)
    _validate_device_compatible(vertices=vertices, faces=faces, texture=texture)
    assert int(faces.max().item()) < int(vertices.shape[0]), (
        "Expected `faces` indices to reference existing vertices only. "
        f"{int(faces.max().item())=} {int(vertices.shape[0])=}"
    )

    if texture is None:
        return

    assert isinstance(texture, MeshTexture), (
        "Expected `texture` to be `None` or a `MeshTexture` instance. "
        f"{type(texture)=}"
    )

    if isinstance(texture, MeshTextureVertexColor):
        assert int(texture.vertex_color.shape[0]) == int(vertices.shape[0]), (
            "Expected `vertex_color` to align one RGB value per vertex. "
            f"{texture.vertex_color.shape=} {vertices.shape=}"
        )
        return

    if isinstance(texture, MeshTextureUVTextureMap):
        assert texture.face_uvs.shape == faces.shape, (
            "Expected `face_uvs` to align one UV triangle per mesh face. "
            f"{texture.face_uvs.shape=} {faces.shape=}"
        )
        return

    raise AssertionError(
        "Expected `texture` to be a `MeshTextureVertexColor` or "
        "`MeshTextureUVTextureMap`. "
        f"{type(texture)=}"
    )
