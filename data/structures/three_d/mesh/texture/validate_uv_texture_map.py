from typing import Any

import torch


def validate_uv_texture_map(
    uv_texture_map: torch.Tensor,
    verts_uvs: torch.Tensor,
    faces_uvs: torch.Tensor,
    verts_uvs_convention: str,
) -> None:
    """Validate one UV-texture-map representation: single-field validators plus cross-field invariants.

    Args:
        uv_texture_map: UV texture image (CHW/HWC/NCHW/NHWC, 3 channels;
            uint8 [0, 255] or float32 [0, 1]).
        verts_uvs: UV-coordinate table [U, 2], float, finite, non-negative;
            values may exceed 1 per the seam-safe canonical contract on
            `MeshTextureUVTextureMap`.
        faces_uvs: Face-to-UV index tensor [F, 3], integer, non-empty,
            non-negative indices.
        verts_uvs_convention: UV-origin convention string (`"obj"` or
            `"top_left"`).

    Returns:
        None.
    """

    validate_uv_texture_map_image(obj=uv_texture_map)
    validate_verts_uvs(obj=verts_uvs)
    validate_faces_uvs(obj=faces_uvs)
    validate_convention(obj=verts_uvs_convention)
    _validate_verts_uvs_faces_uvs_cross_field(
        verts_uvs=verts_uvs,
        faces_uvs=faces_uvs,
    )


def validate_uv_texture_map_image(obj: Any) -> None:
    """Validate a UV texture image tensor (HWC/CHW/NHWC/NCHW, 3 channels; uint8 or float32).

    Args:
        obj: Candidate UV texture image.

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
    assert 0, (
        "Expected `uv_texture_map` to be either uint8 `[0, 255]` or "
        "float32 `[0, 1]`. "
        f"{obj.dtype=}"
    )


def validate_verts_uvs(obj: Any) -> None:
    """Validate a UV-coordinate table (float [U, 2], finite, non-negative; values may exceed 1 per the seam-safe canonical contract on `MeshTextureUVTextureMap`).

    Args:
        obj: Candidate UV-coordinate table.

    Returns:
        None.
    """

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


def validate_faces_uvs(obj: Any) -> None:
    """Validate a face-to-UV index tensor (integer [F, 3], non-empty, non-negative indices).

    Args:
        obj: Candidate face-to-UV index tensor.

    Returns:
        None.
    """

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


def validate_convention(obj: Any) -> str:
    """Validate and return a UV-origin convention string (one of `"obj"`, `"top_left"`).

    Args:
        obj: Candidate UV-origin convention string.

    Returns:
        The validated UV-origin convention string `obj`.
    """

    assert isinstance(obj, str), (
        "Expected `verts_uvs_convention` to be a string. " f"{type(obj)=}"
    )
    assert obj in ("obj", "top_left"), "Unsupported mesh UV convention. " f"{obj=}"
    return obj


def _validate_verts_uvs_faces_uvs_cross_field(
    verts_uvs: torch.Tensor,
    faces_uvs: torch.Tensor,
) -> None:
    """Validate the cross-field invariants between verts_uvs and faces_uvs.

    Args:
        verts_uvs: UV-coordinate table [U, 2].
        faces_uvs: Face-to-UV index tensor [F, 3].

    Returns:
        None.
    """

    def _validate_faces_uvs_index_range() -> None:
        assert int(faces_uvs.max().item()) < int(verts_uvs.shape[0]), (
            "Expected `faces_uvs` indices to reference existing `verts_uvs` "
            "rows only. "
            f"{int(faces_uvs.max().item())=} {int(verts_uvs.shape[0])=}"
        )

    _validate_faces_uvs_index_range()

    def _validate_seam_safe_uv_layout() -> None:
        face_corner_u = verts_uvs[faces_uvs.to(dtype=torch.long), 0]
        sorted_face_corner_u = face_corner_u.sort(dim=1).values
        interior_gaps = sorted_face_corner_u[:, 1:] - sorted_face_corner_u[:, :-1]
        wraparound_gap = sorted_face_corner_u[:, 0] + 1.0 - sorted_face_corner_u[:, -1]
        largest_interior_gap = interior_gaps.max(dim=1).values
        worst_face_interior_excess = float(
            (largest_interior_gap - wraparound_gap).max().item()
        )
        assert worst_face_interior_excess <= 0.0, (
            "Expected every face to be in non-wrapping canonical form: its "
            "corners contiguous, so the wraparound gap (min_u + 1 - max_u) is "
            ">= every interior gap between the sorted corner-u's. A face whose "
            "largest cyclic gap is an interior gap straddles the cylindrical "
            "wrap without having been seam-shifted into canonical form. "
            f"{worst_face_interior_excess=}"
        )

    _validate_seam_safe_uv_layout()


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
