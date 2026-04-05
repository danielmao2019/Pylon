from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from data.structures.three_d.mesh.conventions import transform_vertex_uv_convention
from data.structures.three_d.mesh.load import load_mesh
from data.structures.three_d.mesh.save import save_mesh
from data.structures.three_d.mesh.validate import (
    validate_mesh_attributes,
    validate_mesh_uv_convention,
    validate_uv_texture_map,
    validate_vertex_color,
)


def _resolve_texture_mode(
    vertex_uv: Optional[torch.Tensor],
) -> str:
    """Resolve one mesh texture mode from its attributes.

    Args:
        vertex_uv: Optional UV-coordinate table `[U, 2]`.

    Returns:
        `uv_texture_map` when UV coordinates are present, otherwise `vertex_color`.
    """

    def _validate_inputs() -> None:
        """Validate one texture-mode input set.

        Args:
            None.

        Returns:
            None.
        """

        assert vertex_uv is None or isinstance(vertex_uv, torch.Tensor), (
            "Expected `vertex_uv` to be `None` or a tensor. " f"{type(vertex_uv)=}"
        )

    _validate_inputs()

    if vertex_uv is None:
        return "vertex_color"
    return "uv_texture_map"


class Mesh:
    """Store one triangle mesh with optional texture attributes.

    Args:
        vertices: Mesh vertex tensor `[V, 3]`.
        faces: Mesh face tensor `[F, 3]`.
        vertex_color: Optional per-vertex RGB tensor `[V, 3]`.
        uv_texture_map: Optional UV texture image.
        vertex_uv: Optional UV-coordinate table `[U, 2]`.
        face_uvs: Optional UV-face indices `[F, 3]`.
        convention: Optional UV-origin convention for `vertex_uv`. `obj` means
            `v=0` is the bottom edge. `top_left` means `v=0` is the top edge.

    Returns:
        None.
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        vertex_color: Optional[torch.Tensor] = None,
        uv_texture_map: Optional[torch.Tensor] = None,
        vertex_uv: Optional[torch.Tensor] = None,
        face_uvs: Optional[torch.Tensor] = None,
        convention: Optional[str] = None,
    ) -> None:
        """Initialize one mesh container.

        Args:
            vertices: Mesh vertex tensor `[V, 3]`.
            faces: Mesh face tensor `[F, 3]`.
            vertex_color: Optional per-vertex RGB tensor `[V, 3]`.
            uv_texture_map: Optional UV texture image.
            vertex_uv: Optional UV-coordinate table `[U, 2]`.
            face_uvs: Optional UV-face indices `[F, 3]`.
            convention: Optional UV-origin convention for `vertex_uv`.

        Returns:
            None.
        """

        def _validate_inputs() -> None:
            """Validate one mesh-construction input set.

            Args:
                None.

            Returns:
                None.
            """
            validate_mesh_attributes(
                vertices=vertices,
                faces=faces,
                vertex_color=vertex_color,
                uv_texture_map=uv_texture_map,
                vertex_uv=vertex_uv,
                face_uvs=face_uvs,
                convention=convention,
            )

        _validate_inputs()

        def _normalize_inputs() -> Tuple[
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[str],
        ]:
            return (
                vertices.contiguous(),
                faces.to(dtype=torch.int64).contiguous(),
                (
                    None
                    if vertex_color is None
                    else Mesh.normalize_vertex_color(vertex_color=vertex_color)
                ),
                (
                    None
                    if uv_texture_map is None
                    else Mesh.normalize_uv_texture_map(uv_texture_map=uv_texture_map)
                ),
                None if vertex_uv is None else vertex_uv.contiguous(),
                (
                    None
                    if face_uvs is None
                    else face_uvs.to(dtype=torch.int64).contiguous()
                ),
                convention,
            )

        (
            vertices,
            faces,
            vertex_color,
            uv_texture_map,
            vertex_uv,
            face_uvs,
            convention,
        ) = _normalize_inputs()

        self.vertices = vertices
        self.faces = faces
        self.vertex_color = vertex_color
        self.uv_texture_map = uv_texture_map
        self.vertex_uv = vertex_uv
        self.face_uvs = face_uvs
        self.convention = convention
        self.device = self.vertices.device
        self.texture_mode = _resolve_texture_mode(vertex_uv=self.vertex_uv)

    @staticmethod
    def normalize_vertex_color(
        vertex_color: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize one vertex-color tensor to the mesh canonical format.

        Args:
            vertex_color: Vertex colors in `[V, 3]` or `[1, V, 3]` layout with
                uint8 `[0, 255]` or float32 `[0, 1]` values.

        Returns:
            Vertex colors in contiguous float32 `[V, 3]` layout with values
            clamped to `[0, 1]`.
        """
        validate_vertex_color(obj=vertex_color)

        if vertex_color.ndim == 3:
            vertex_color = vertex_color[0]
        vertex_color = vertex_color.contiguous()
        if vertex_color.dtype == torch.uint8:
            return (
                vertex_color.to(dtype=torch.float32)
                .div(255.0)
                .clamp(0.0, 1.0)
                .contiguous()
            )
        return vertex_color.to(dtype=torch.float32).clamp(0.0, 1.0).contiguous()

    @staticmethod
    def normalize_uv_texture_map(
        uv_texture_map: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize one UV texture map to the mesh canonical format.

        Args:
            uv_texture_map: UV texture map in CHW, HWC, NCHW, or NHWC layout with
                uint8 `[0, 255]` or float32 `[0, 1]` values.

        Returns:
            UV texture map in contiguous float32 HWC layout with values clamped to
            `[0, 1]`.
        """
        validate_uv_texture_map(obj=uv_texture_map)

        if uv_texture_map.ndim == 4:
            uv_texture_map = uv_texture_map[0]
        if uv_texture_map.shape[0] == 3:
            uv_texture_map = uv_texture_map.permute(1, 2, 0)
        uv_texture_map = uv_texture_map.contiguous()
        if uv_texture_map.dtype == torch.uint8:
            return (
                uv_texture_map.to(dtype=torch.float32)
                .div(255.0)
                .clamp(0.0, 1.0)
                .contiguous()
            )
        return uv_texture_map.to(dtype=torch.float32).clamp(0.0, 1.0).contiguous()

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Mesh":
        """Load one mesh from disk.

        Args:
            path: Mesh file path or supported mesh-root directory path.

        Returns:
            Loaded mesh instance.
        """

        def _validate_inputs() -> None:
            assert isinstance(path, (str, Path)), (
                "Expected `path` to be a `str` or `Path`. " f"{type(path)=}"
            )

        _validate_inputs()

        mesh_attributes = load_mesh(path=path)
        return cls(**mesh_attributes)

    def save(self, path: Union[str, Path]) -> None:
        """Save one mesh to disk.

        Args:
            path: Output OBJ path or output directory path.

        Returns:
            None.
        """

        def _validate_inputs() -> None:
            assert isinstance(path, (str, Path)), (
                "Expected `path` to be a `str` or `Path`. " f"{type(path)=}"
            )

        _validate_inputs()

        save_mesh(mesh=self, output_path=path)

    def to(
        self,
        device: Union[str, torch.device, None] = None,
        convention: Optional[str] = None,
    ) -> "Mesh":
        """Move one mesh to a target device and/or UV convention.

        Args:
            device: Optional target device.
            convention: Optional target UV-origin convention.

        Returns:
            This mesh when the device and convention already match, otherwise a new
            mesh on the requested target.
        """

        def _validate_inputs() -> None:
            assert device is None or isinstance(device, (str, torch.device)), (
                "Expected `device` to be `None`, a `str`, or a `torch.device`. "
                f"{type(device)=}"
            )
            assert convention is None or isinstance(convention, str), (
                "Expected `convention` to be `None` or a string. "
                f"{type(convention)=}"
            )
            if convention is not None:
                validate_mesh_uv_convention(convention=convention)
                assert self.convention is not None, (
                    "Expected only UV meshes to support explicit UV-convention "
                    "conversion. "
                    f"{self.convention=} {convention=}"
                )

        _validate_inputs()

        target_device = self.device if device is None else torch.device(device)
        target_convention = self.convention if convention is None else convention
        if self.device == target_device and self.convention == target_convention:
            return self

        target_vertex_uv = self.vertex_uv
        if target_convention != self.convention:
            target_vertex_uv = transform_vertex_uv_convention(
                vertex_uv=self.vertex_uv,
                source_convention=self.convention,
                target_convention=target_convention,
            )

        return Mesh(
            vertices=self.vertices.to(device=target_device),
            faces=self.faces.to(device=target_device),
            vertex_color=(
                None
                if self.vertex_color is None
                else self.vertex_color.to(device=target_device)
            ),
            uv_texture_map=(
                None
                if self.uv_texture_map is None
                else self.uv_texture_map.to(device=target_device)
            ),
            vertex_uv=(
                None
                if target_vertex_uv is None
                else target_vertex_uv.to(device=target_device)
            ),
            face_uvs=(
                None
                if self.face_uvs is None
                else self.face_uvs.to(device=target_device)
            ),
            convention=target_convention,
        )
