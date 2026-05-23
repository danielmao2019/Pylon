from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from data.structures.three_d.mesh.texture.mesh_texture import MeshTexture
from data.structures.three_d.mesh.validate import validate_mesh_attributes


class Mesh:
    """One triangle mesh: geometry plus an optional texture.

    `verts` / `faces` always hold the geometry domain — the distinct surface
    positions and the faces indexing them. `texture is None` means the mesh is
    geometry-only.

    Args:
        verts: Mesh vertex tensor `[V, 3]`.
        faces: Mesh face tensor `[F, 3]`.
        texture: Optional `MeshTexture` (`MeshTextureVertexColor` or
            `MeshTextureUVTextureMap`).

    Returns:
        None.
    """

    verts: torch.Tensor
    faces: torch.Tensor
    texture: Optional[MeshTexture]
    device: torch.device

    def __init__(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        texture: Optional[MeshTexture] = None,
    ) -> None:
        """Initialize one mesh container.

        Args:
            verts: Mesh vertex tensor `[V, 3]`.
            faces: Mesh face tensor `[F, 3]`.
            texture: Optional `MeshTexture`.

        Returns:
            None.
        """

        def _validate_inputs() -> None:
            validate_mesh_attributes(verts=verts, faces=faces, texture=texture)

        _validate_inputs()

        def _normalize_inputs() -> Tuple[torch.Tensor, torch.Tensor]:
            return (
                verts.contiguous(),
                faces.to(dtype=torch.int64).contiguous(),
            )

        verts, faces = _normalize_inputs()

        self.verts = verts
        self.faces = faces
        self.texture = texture
        self.device = self.verts.device

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Mesh":
        """Load one mesh from an OBJ file or a mesh-root directory.

        Args:
            path: Mesh file path or supported mesh-root directory path.

        Returns:
            Loaded mesh instance.
        """

        from data.structures.three_d.mesh.load import load_mesh

        return load_mesh(path=path)

    def save(self, path: Union[str, Path]) -> None:
        """Save this mesh to an OBJ/PLY file or a directory.

        Args:
            path: Output OBJ/PLY path or output directory path.

        Returns:
            None.
        """

        from data.structures.three_d.mesh.save import save_mesh

        save_mesh(mesh=self, output_path=path)

    def to(
        self,
        device: Union[str, torch.device, None] = None,
        convention: Optional[str] = None,
    ) -> "Mesh":
        """Return this mesh on a target device and/or UV-origin convention.

        Args:
            device: Optional target device.
            convention: Optional target UV-origin convention.

        Returns:
            This mesh when the device and convention already match, otherwise a
            new mesh on the requested target.
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
                assert self.texture is not None, (
                    "Expected only textured meshes to support explicit "
                    "UV-convention conversion. "
                    f"{self.texture=} {convention=}"
                )

        _validate_inputs()

        target_device = self.device if device is None else torch.device(device)
        if self.device == target_device and convention is None:
            return self

        if self.texture is None:
            return Mesh(
                verts=self.verts.to(device=target_device),
                faces=self.faces.to(device=target_device),
                texture=None,
            )

        return Mesh(
            verts=self.verts.to(device=target_device),
            faces=self.faces.to(device=target_device),
            texture=self.texture.to(device=target_device, convention=convention),
        )
