from typing import Any

import torch


def validate_uv_texture_map(
    uv_texture_map: torch.Tensor,
    vertex_uv: torch.Tensor,
    face_uvs: torch.Tensor,
    convention: str,
) -> None:
    """Validate the whole uv-texture-map representation.

    The module's single public API. Validates every field plus the cross-field
    invariant that `face_uvs` indices reference valid `vertex_uv` rows.

    Args:
        uv_texture_map: Candidate UV texture map in HWC, CHW, NHWC, or NCHW
            layout with uint8 `[0, 255]` or float32 `[0, 1]` values.
        vertex_uv: Candidate UV-coordinate table `[U, 2]`.
        face_uvs: Candidate face-to-UV index tensor `[F, 3]`.
        convention: Candidate UV-origin convention string.

    Returns:
        None.
    """

    _validate_uv_texture_map_image(obj=uv_texture_map)
    _validate_vertex_uv(obj=vertex_uv)
    _validate_face_uvs(obj=face_uvs)
    _validate_mesh_uv_convention(convention=convention)

    assert int(face_uvs.max().item()) < int(vertex_uv.shape[0]), (
        "Expected `face_uvs` indices to reference existing `vertex_uv` rows only. "
        f"{int(face_uvs.max().item())=} {int(vertex_uv.shape[0])=}"
    )


def _validate_uv_texture_map_image(obj: Any) -> None:
    """Validate one UV texture image tensor.

    Args:
        obj: Candidate UV texture map in HWC, CHW, NHWC, or NCHW layout.

    Returns:
        None.
    """

    assert isinstance(obj, torch.Tensor), (
        "Expected `uv_texture_map` to be a `torch.Tensor`. " f"{type(obj)=}"
    )
    assert obj.ndim in (3, 4), (
        "Expected `uv_texture_map` to be rank 3 or 4. " f"{obj.shape=}"
    )
    if obj.ndim == 3:
        assert obj.shape[0] == 3 or obj.shape[2] == 3, (
            "Expected rank-3 `uv_texture_map` to be CHW or HWC with 3 channels. "
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


def _validate_vertex_uv(obj: Any) -> None:
    """Validate one UV-coordinate table.

    Args:
        obj: Candidate UV-coordinate tensor with shape `[U, 2]`.

    Returns:
        None.
    """

    assert isinstance(obj, torch.Tensor), (
        "Expected `vertex_uv` to be a `torch.Tensor`. " f"{type(obj)=}"
    )
    assert obj.ndim == 2, "Expected `vertex_uv` to be rank 2. " f"{obj.shape=}"
    assert obj.shape[1] == 2, (
        "Expected `vertex_uv` to contain UV pairs in the last dimension. "
        f"{obj.shape=}"
    )
    assert obj.shape[0] > 0, (
        "Expected `vertex_uv` to contain at least one UV coordinate. " f"{obj.shape=}"
    )
    assert obj.is_floating_point(), (
        "Expected `vertex_uv` to use a floating dtype. " f"{obj.dtype=}"
    )
    assert torch.isfinite(obj).all(), (
        "Expected `vertex_uv` to contain only finite values. "
        f"{obj.shape=} {obj.dtype=}"
    )
    assert float(obj.min().item()) >= 0.0, (
        "Expected `vertex_uv` values to be at least 0. " f"{float(obj.min().item())=}"
    )
    assert float(obj.max().item()) <= 1.0, (
        "Expected `vertex_uv` values to be at most 1. " f"{float(obj.max().item())=}"
    )


def _validate_face_uvs(obj: Any) -> None:
    """Validate one face-to-UV index tensor.

    Args:
        obj: Candidate UV-face tensor with shape `[F, 3]`.

    Returns:
        None.
    """

    assert isinstance(obj, torch.Tensor), (
        "Expected `face_uvs` to be a `torch.Tensor`. " f"{type(obj)=}"
    )
    assert obj.ndim == 2, "Expected `face_uvs` to be rank 2. " f"{obj.shape=}"
    assert obj.shape[1] == 3, (
        "Expected `face_uvs` to contain triangular UV indices. " f"{obj.shape=}"
    )
    assert obj.shape[0] > 0, (
        "Expected `face_uvs` to contain at least one face. " f"{obj.shape=}"
    )
    assert not obj.is_floating_point(), (
        "Expected `face_uvs` to use an integer dtype. " f"{obj.dtype=}"
    )
    assert obj.dtype != torch.bool, (
        "Expected `face_uvs` to use an integer index dtype, not bool. " f"{obj.dtype=}"
    )
    assert int(obj.min().item()) >= 0, (
        "Expected `face_uvs` to contain only non-negative indices. "
        f"{int(obj.min().item())=}"
    )


def _validate_mesh_uv_convention(convention: Any) -> str:
    """Validate and return one mesh UV-origin convention.

    Args:
        convention: Candidate UV-origin convention string.

    Returns:
        Validated convention string.
    """

    assert isinstance(convention, str), (
        "Expected `convention` to be a string. " f"{type(convention)=}"
    )
    assert convention in [
        "obj",
        "top_left",
    ], (
        "Unsupported mesh UV convention. " f"{convention=}"
    )
    return convention


def _validate_uv_texture_map_image_uint8(obj: Any) -> None:
    """Validate one uint8 UV texture image tensor.

    Args:
        obj: Candidate UV texture tensor.

    Returns:
        None.
    """

    assert obj.dtype == torch.uint8, (
        "Expected `uv_texture_map` uint8 validation to receive uint8 values. "
        f"{obj.dtype=}"
    )


def _validate_uv_texture_map_image_float32(obj: Any) -> None:
    """Validate one float32 UV texture image tensor.

    Args:
        obj: Candidate UV texture tensor.

    Returns:
        None.
    """

    assert obj.dtype == torch.float32, (
        "Expected `uv_texture_map` float32 validation to receive float32 values. "
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
