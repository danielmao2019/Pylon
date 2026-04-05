from typing import Any, Optional

import torch


def validate_mesh_uv_convention(convention: Any) -> str:
    """Validate one mesh UV-origin convention.

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


def validate_vertex_color(obj: Any) -> None:
    """Validate one per-vertex RGB tensor.

    Args:
        obj: Candidate vertex-color tensor with shape `[V, 3]` or `[1, V, 3]`.

    Returns:
        None.
    """

    assert isinstance(obj, torch.Tensor), (
        "Expected `vertex_color` to be a `torch.Tensor`. " f"{type(obj)=}"
    )
    assert obj.ndim in (2, 3), (
        "Expected `vertex_color` to be rank 2 or 3. " f"{obj.shape=}"
    )
    if obj.ndim == 3:
        assert obj.shape[0] == 1, (
            "Expected rank-3 `vertex_color` to have batch size 1. " f"{obj.shape=}"
        )
        obj = obj[0]
    assert obj.shape[1] == 3, (
        "Expected `vertex_color` to have RGB values in the last dimension. "
        f"{obj.shape=}"
    )
    assert obj.shape[0] > 0, (
        "Expected `vertex_color` to contain at least one vertex. " f"{obj.shape=}"
    )

    try:
        _validate_vertex_color_uint8(obj=obj)
        return
    except AssertionError as uint8_error:
        try:
            _validate_vertex_color_float32(obj=obj)
            return
        except AssertionError as float32_error:
            raise AssertionError(
                "Expected `vertex_color` to be either uint8 `[0, 255]` or "
                "float32 `[0, 1]`. "
                f"{obj.dtype=} {uint8_error=} {float32_error=}"
            ) from float32_error


def validate_uv_texture_map(obj: Any) -> None:
    """Validate one UV texture map tensor.

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
            "Expected rank-4 `uv_texture_map` to be NCHW or NHWC with 3 channels. "
            f"{obj.shape=}"
        )
        texture_height = int(obj.shape[2] if obj.shape[1] == 3 else obj.shape[1])
        texture_width = int(obj.shape[3] if obj.shape[1] == 3 else obj.shape[2])
    assert texture_height > 0 and texture_width > 0, (
        "Expected `uv_texture_map` to have positive spatial resolution. "
        f"{obj.shape=}"
    )

    try:
        _validate_uv_texture_map_uint8(obj=obj)
        return
    except AssertionError as uint8_error:
        try:
            _validate_uv_texture_map_float32(obj=obj)
            return
        except AssertionError as float32_error:
            raise AssertionError(
                "Expected `uv_texture_map` to be either uint8 `[0, 255]` or "
                "float32 `[0, 1]`. "
                f"{obj.dtype=} {uint8_error=} {float32_error=}"
            ) from float32_error


def _validate_vertex_color_uint8(obj: Any) -> None:
    """Validate one uint8 vertex-color tensor.

    Args:
        obj: Candidate vertex-color tensor.

    Returns:
        None.
    """

    assert obj.dtype == torch.uint8, (
        "Expected `vertex_color` uint8 validation to receive uint8 values. "
        f"{obj.dtype=}"
    )


def _validate_vertex_color_float32(obj: Any) -> None:
    """Validate one float32 vertex-color tensor.

    Args:
        obj: Candidate vertex-color tensor.

    Returns:
        None.
    """

    assert obj.dtype == torch.float32, (
        "Expected `vertex_color` float32 validation to receive float32 values. "
        f"{obj.dtype=}"
    )
    assert torch.isfinite(obj).all(), (
        "Expected float32 `vertex_color` to contain only finite values. "
        f"{obj.shape=} {obj.dtype=}"
    )
    assert float(obj.min().item()) >= 0.0, (
        "Expected float32 `vertex_color` values to be at least 0. "
        f"{float(obj.min().item())=}"
    )
    assert float(obj.max().item()) <= 1.0, (
        "Expected float32 `vertex_color` values to be at most 1. "
        f"{float(obj.max().item())=}"
    )


def _validate_uv_texture_map_uint8(obj: Any) -> None:
    """Validate one uint8 UV texture map tensor.

    Args:
        obj: Candidate UV texture tensor.

    Returns:
        None.
    """

    assert obj.dtype == torch.uint8, (
        "Expected `uv_texture_map` uint8 validation to receive uint8 values. "
        f"{obj.dtype=}"
    )


def _validate_uv_texture_map_float32(obj: Any) -> None:
    """Validate one float32 UV texture map tensor.

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
        "Expected float32 `uv_texture_map` to contain only finite values. "
        f"{obj.shape=} {obj.dtype=}"
    )
    assert float(obj.min().item()) >= 0.0, (
        "Expected float32 `uv_texture_map` values to be at least 0. "
        f"{float(obj.min().item())=}"
    )
    assert float(obj.max().item()) <= 1.0, (
        "Expected float32 `uv_texture_map` values to be at most 1. "
        f"{float(obj.max().item())=}"
    )


def validate_vertex_uv(obj: Any) -> None:
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


def validate_face_uvs(obj: Any) -> None:
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


def _validate_device_compatible(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_color: Optional[torch.Tensor] = None,
    uv_texture_map: Optional[torch.Tensor] = None,
    vertex_uv: Optional[torch.Tensor] = None,
    face_uvs: Optional[torch.Tensor] = None,
) -> None:
    """Validate that all provided mesh tensors live on one device.

    Args:
        vertices: Mesh vertex tensor `[V, 3]`.
        faces: Mesh face tensor `[F, 3]`.
        vertex_color: Optional per-vertex RGB tensor `[V, 3]`.
        uv_texture_map: Optional UV texture tensor.
        vertex_uv: Optional UV-coordinate table `[U, 2]`.
        face_uvs: Optional UV-face indices `[F, 3]`.

    Returns:
        None.
    """

    assert faces.device == vertices.device, (
        "Expected `faces` to live on the same device as `vertices`. "
        f"{faces.device=} {vertices.device=}"
    )
    if vertex_color is not None:
        assert vertex_color.device == vertices.device, (
            "Expected `vertex_color` to live on the same device as `vertices`. "
            f"{vertex_color.device=} {vertices.device=}"
        )
    if vertex_uv is not None:
        assert vertex_uv.device == vertices.device, (
            "Expected `vertex_uv` to live on the same device as `vertices`. "
            f"{vertex_uv.device=} {vertices.device=}"
        )
    if face_uvs is not None:
        assert face_uvs.device == vertices.device, (
            "Expected `face_uvs` to live on the same device as `vertices`. "
            f"{face_uvs.device=} {vertices.device=}"
        )
    if uv_texture_map is not None:
        assert uv_texture_map.device == vertices.device, (
            "Expected `uv_texture_map` to live on the same device as `vertices`. "
            f"{uv_texture_map.device=} {vertices.device=}"
        )


def validate_mesh_attributes(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_color: Optional[torch.Tensor] = None,
    uv_texture_map: Optional[torch.Tensor] = None,
    vertex_uv: Optional[torch.Tensor] = None,
    face_uvs: Optional[torch.Tensor] = None,
    convention: Optional[str] = None,
) -> None:
    """Validate the full attribute set for one mesh instance.

    Args:
        vertices: Mesh vertex tensor `[V, 3]`.
        faces: Mesh face tensor `[F, 3]`.
        vertex_color: Optional per-vertex RGB tensor `[V, 3]`.
        uv_texture_map: Optional UV texture tensor.
        vertex_uv: Optional UV-coordinate table `[U, 2]`.
        face_uvs: Optional UV-face indices `[F, 3]`.
        convention: Optional UV-origin convention for `vertex_uv`.

    Returns:
        None.
    """

    validate_vertices(obj=vertices)
    validate_faces(obj=faces)
    _validate_device_compatible(
        vertices=vertices,
        faces=faces,
        vertex_color=vertex_color,
        uv_texture_map=uv_texture_map,
        vertex_uv=vertex_uv,
        face_uvs=face_uvs,
    )
    assert int(faces.max().item()) < int(vertices.shape[0]), (
        "Expected `faces` indices to reference existing vertices only. "
        f"{int(faces.max().item())=} {int(vertices.shape[0])=}"
    )

    has_vertex_color = vertex_color is not None
    has_uv_texture_map = uv_texture_map is not None
    assert not (has_vertex_color and has_uv_texture_map), (
        "Expected mesh attributes to avoid carrying both texture representations "
        "at once. "
        f"{has_vertex_color=} {has_uv_texture_map=}"
    )

    if vertex_color is not None:
        validate_vertex_color(obj=vertex_color)
        assert vertex_color.ndim == 2, (
            "Expected stored mesh `vertex_color` to be canonical rank 2 `[V, 3]`. "
            f"{vertex_color.shape=}"
        )
        assert int(vertex_color.shape[0]) == int(vertices.shape[0]), (
            "Expected `vertex_color` to align one RGB value per vertex. "
            f"{vertex_color.shape=} {vertices.shape=}"
        )

    has_vertex_uv = vertex_uv is not None
    has_face_uvs = face_uvs is not None
    assert has_vertex_uv == has_face_uvs, (
        "Expected UV topology to provide both `vertex_uv` and `face_uvs`, or neither. "
        f"{has_vertex_uv=} {has_face_uvs=}"
    )
    if has_vertex_uv:
        assert convention is not None, (
            "Expected UV meshes to specify a `convention` for `vertex_uv`. "
            f"{convention=}"
        )
        validate_mesh_uv_convention(convention=convention)
        validate_vertex_uv(obj=vertex_uv)
        validate_face_uvs(obj=face_uvs)
        assert (
            face_uvs is not None
        ), "Expected `face_uvs` to be present when UVs are used."
        assert (
            vertex_uv is not None
        ), "Expected `vertex_uv` to be present when UVs are used."
        assert face_uvs.shape == faces.shape, (
            "Expected `face_uvs` to align one UV triangle per mesh face. "
            f"{face_uvs.shape=} {faces.shape=}"
        )
        assert int(face_uvs.max().item()) < int(vertex_uv.shape[0]), (
            "Expected `face_uvs` indices to reference existing UV coordinates only. "
            f"{int(face_uvs.max().item())=} {int(vertex_uv.shape[0])=}"
        )
    else:
        assert convention is None, (
            "Expected non-UV meshes to leave `convention` unset. " f"{convention=}"
        )

    if uv_texture_map is not None:
        validate_uv_texture_map(obj=uv_texture_map)
        assert has_vertex_uv, (
            "Expected UV-textured meshes to include `vertex_uv` and `face_uvs`. "
            f"{has_vertex_uv=} {has_face_uvs=}"
        )
