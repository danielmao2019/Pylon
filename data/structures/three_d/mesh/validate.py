from typing import Any, Optional

import torch

from data.structures.three_d.mesh.texture.mesh_texture import MeshTexture
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)


def validate_verts(obj: Any) -> None:
    assert isinstance(obj, torch.Tensor), (
        "Expected `verts` to be a `torch.Tensor`. " f"{type(obj)=}"
    )
    assert obj.ndim == 2, "Expected `verts` to be rank 2. " f"{obj.shape=}"
    assert obj.shape[1] == 3, (
        "Expected `verts` to have XYZ coordinates in the last dimension. "
        f"{obj.shape=}"
    )
    assert obj.shape[0] > 0, (
        "Expected `verts` to contain at least one vertex. " f"{obj.shape=}"
    )
    assert obj.is_floating_point(), (
        "Expected `verts` to use a floating dtype. " f"{obj.dtype=}"
    )
    assert torch.isfinite(obj).all(), (
        "Expected `verts` to contain only finite values. "
        f"{obj.shape=} {obj.dtype=}"
    )


def validate_faces(obj: Any) -> None:
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
    verts: torch.Tensor,
    faces: torch.Tensor,
    texture: Optional[MeshTexture],
) -> None:
    assert faces.device == verts.device, (
        "Expected `faces` to live on the same device as `verts`. "
        f"{faces.device=} {verts.device=}"
    )
    if texture is not None:
        assert texture.device == verts.device, (
            "Expected the mesh texture to live on the same device as "
            "`verts`. "
            f"{texture.device=} {verts.device=}"
        )


def validate_mesh_attributes(
    verts: torch.Tensor,
    faces: torch.Tensor,
    texture: Optional[MeshTexture] = None,
) -> None:
    validate_verts(obj=verts)
    validate_faces(obj=faces)
    _validate_device_compatible(verts=verts, faces=faces, texture=texture)
    assert int(faces.max().item()) < int(verts.shape[0]), (
        "Expected `faces` indices to reference existing verts only. "
        f"{int(faces.max().item())=} {int(verts.shape[0])=}"
    )

    if texture is None:
        return

    assert isinstance(texture, MeshTexture), (
        "Expected `texture` to be `None` or a `MeshTexture` instance. "
        f"{type(texture)=}"
    )
    if isinstance(texture, MeshTextureVertexColor):
        assert int(texture.vertex_color.shape[0]) == int(verts.shape[0]), (
            "Expected `vertex_color` to align one RGB value per vertex. "
            f"{texture.vertex_color.shape=} {verts.shape=}"
        )
        return
    if isinstance(texture, MeshTextureUVTextureMap):
        assert texture.faces_uvs.shape == faces.shape, (
            "Expected `faces_uvs` to align one UV triangle per mesh face. "
            f"{texture.faces_uvs.shape=} {faces.shape=}"
        )
        return
    raise AssertionError(
        "Expected `texture` to be a `MeshTextureVertexColor` or "
        "`MeshTextureUVTextureMap`. "
        f"{type(texture)=}"
    )
