from typing import Any

import torch


def validate_uv_texture_map(
    uv_texture_map: torch.Tensor,
    verts_uvs: torch.Tensor,
    faces_uvs: torch.Tensor,
    convention: str,
) -> None:
    _validate_uv_texture_map_image(obj=uv_texture_map)
    _validate_verts_uvs(obj=verts_uvs)
    _validate_faces_uvs(obj=faces_uvs)
    _validate_mesh_uv_convention(convention=convention)

    assert int(faces_uvs.max().item()) < int(verts_uvs.shape[0]), (
        "Expected `faces_uvs` indices to reference existing `verts_uvs` rows "
        "only. "
        f"{int(faces_uvs.max().item())=} {int(verts_uvs.shape[0])=}"
    )


def _validate_uv_texture_map_image(obj: Any) -> None:
    assert isinstance(obj, torch.Tensor), (
        "Expected `uv_texture_map` to be a `torch.Tensor`. " f"{type(obj)=}"
    )
    assert obj.ndim in (3, 4), (
        "Expected `uv_texture_map` to be rank 3 or 4. " f"{obj.shape=}"
    )
    if obj.ndim == 3:
        assert obj.shape[0] == 3 or obj.shape[2] == 3, (
            "Expected rank-3 `uv_texture_map` to be CHW or HWC with 3 "
            "channels. "
            f"{obj.shape=}"
        )
        texture_height = int(obj.shape[1] if obj.shape[0] == 3 else obj.shape[0])
        texture_width = int(obj.shape[2] if obj.shape[0] == 3 else obj.shape[1])
    else:
        assert obj.shape[0] == 1, (
            "Expected rank-4 `uv_texture_map` to have batch size 1. " f"{obj.shape=}"
        )
        assert obj.shape[1] == 3 or obj.shape[3] == 3, (
            "Expected rank-4 `uv_texture_map` to be NCHW or NHWC with 3 "
            "channels. "
            f"{obj.shape=}"
        )
        texture_height = int(obj.shape[2] if obj.shape[1] == 3 else obj.shape[1])
        texture_width = int(obj.shape[3] if obj.shape[1] == 3 else obj.shape[2])
    assert texture_height > 0 and texture_width > 0, (
        "Expected `uv_texture_map` to have positive spatial resolution. "
        f"{obj.shape=}"
    )

    if obj.dtype == torch.uint8:
        _validate_uv_texture_map_image_uint8(obj=obj)
        return
    if obj.dtype == torch.float32:
        _validate_uv_texture_map_image_float32(obj=obj)
        return
    raise AssertionError(
        "Expected `uv_texture_map` to be either uint8 `[0, 255]` or "
        "float32 `[0, 1]`. "
        f"{obj.dtype=}"
    )


def _validate_verts_uvs(obj: Any) -> None:
    assert isinstance(obj, torch.Tensor), (
        "Expected `verts_uvs` to be a `torch.Tensor`. " f"{type(obj)=}"
    )
    assert obj.ndim == 2, "Expected `verts_uvs` to be rank 2. " f"{obj.shape=}"
    assert obj.shape[1] == 2, (
        "Expected `verts_uvs` to contain UV pairs in the last dimension. "
        f"{obj.shape=}"
    )
    assert obj.shape[0] > 0, (
        "Expected `verts_uvs` to contain at least one UV coordinate. " f"{obj.shape=}"
    )
    assert obj.is_floating_point(), (
        "Expected `verts_uvs` to use a floating dtype. " f"{obj.dtype=}"
    )
    assert torch.isfinite(obj).all(), (
        "Expected `verts_uvs` to contain only finite values. "
        f"{obj.shape=} {obj.dtype=}"
    )
    assert float(obj.min().item()) >= 0.0, (
        "Expected `verts_uvs` values to be at least 0. " f"{float(obj.min().item())=}"
    )
    assert float(obj.max().item()) <= 1.0, (
        "Expected `verts_uvs` values to be at most 1. " f"{float(obj.max().item())=}"
    )


def _validate_faces_uvs(obj: Any) -> None:
    assert isinstance(obj, torch.Tensor), (
        "Expected `faces_uvs` to be a `torch.Tensor`. " f"{type(obj)=}"
    )
    assert obj.ndim == 2, "Expected `faces_uvs` to be rank 2. " f"{obj.shape=}"
    assert obj.shape[1] == 3, (
        "Expected `faces_uvs` to contain triangular UV indices. " f"{obj.shape=}"
    )
    assert obj.shape[0] > 0, (
        "Expected `faces_uvs` to contain at least one face. " f"{obj.shape=}"
    )
    assert not obj.is_floating_point(), (
        "Expected `faces_uvs` to use an integer dtype. " f"{obj.dtype=}"
    )
    assert obj.dtype != torch.bool, (
        "Expected `faces_uvs` to use an integer index dtype, not bool. " f"{obj.dtype=}"
    )
    assert int(obj.min().item()) >= 0, (
        "Expected `faces_uvs` to contain only non-negative indices. "
        f"{int(obj.min().item())=}"
    )


def _validate_mesh_uv_convention(convention: Any) -> None:
    assert isinstance(convention, str), (
        "Expected `convention` to be a string. " f"{type(convention)=}"
    )
    assert convention in ("obj", "top_left"), (
        "Unsupported mesh UV convention. " f"{convention=}"
    )


def _validate_uv_texture_map_image_uint8(obj: Any) -> None:
    assert obj.dtype == torch.uint8, (
        "Expected `uv_texture_map` uint8 validation to receive uint8 values. "
        f"{obj.dtype=}"
    )


def _validate_uv_texture_map_image_float32(obj: Any) -> None:
    assert obj.dtype == torch.float32, (
        "Expected `uv_texture_map` float32 validation to receive float32 "
        "values. "
        f"{obj.dtype=}"
    )
    assert torch.isfinite(obj).all(), (
        "Expected float32 `uv_texture_map` to contain only finite RGB values. "
        f"{obj.shape=} {obj.dtype=}"
    )
    min_value = float(obj.min().item())
    max_value = float(obj.max().item())
    assert min_value >= 0.0, (
        "Expected float32 `uv_texture_map` values to be at least 0. " f"{min_value=}"
    )
    assert max_value <= 1.0, (
        "Expected float32 `uv_texture_map` values to be at most 1. " f"{max_value=}"
    )
