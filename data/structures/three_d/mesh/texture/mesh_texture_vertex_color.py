from typing import Union

import torch

from data.structures.three_d.mesh.texture.mesh_texture import MeshTexture
from data.structures.three_d.mesh.texture.validate_vertex_color import (
    validate_vertex_color,
)


class MeshTextureVertexColor(MeshTexture):
    """Per-vertex RGB texture.

    Holds `vertex_color` `[V, 3]`, aligned 1:1 with the mesh's vertices.

    Args:
        vertex_color: Per-vertex RGB tensor in `[V, 3]` or `[1, V, 3]` layout
            with uint8 `[0, 255]` or float32 `[0, 1]` values.

    Returns:
        None.
    """

    vertex_color: torch.Tensor

    def __init__(self, vertex_color: torch.Tensor) -> None:
        """Initialize one vertex-color texture.

        Args:
            vertex_color: Per-vertex RGB tensor in `[V, 3]` or `[1, V, 3]`
                layout with uint8 `[0, 255]` or float32 `[0, 1]` values.

        Returns:
            None.
        """

        validate_vertex_color(obj=vertex_color)
        self.vertex_color = MeshTextureVertexColor.normalize_vertex_color(
            vertex_color=vertex_color
        )

    @staticmethod
    def normalize_vertex_color(vertex_color: torch.Tensor) -> torch.Tensor:
        """Normalize one vertex-color tensor to the canonical format.

        Args:
            vertex_color: Vertex colors in `[V, 3]` or `[1, V, 3]` layout with
                uint8 `[0, 255]` or float32 `[0, 1]` values.

        Returns:
            Vertex colors in contiguous float32 `[V, 3]` layout with values in
            `[0, 1]`.
        """

        if vertex_color.ndim == 3:
            vertex_color = vertex_color[0]
        vertex_color = vertex_color.contiguous()
        if vertex_color.dtype == torch.uint8:
            return vertex_color.to(dtype=torch.float32).div(255.0).contiguous()
        return vertex_color.contiguous()

    @property
    def device(self) -> torch.device:
        """Return the device `vertex_color` lives on.

        Args:
            None.

        Returns:
            The `torch.device` `vertex_color` lives on.
        """

        return self.vertex_color.device

    def to(
        self,
        device: Union[str, torch.device, None] = None,
        convention: Union[str, None] = None,
    ) -> "MeshTextureVertexColor":
        """Return this texture on a target device.

        Args:
            device: Optional target device.
            convention: Must be `None`; vertex color carries no UV convention.

        Returns:
            This texture when the device already matches, otherwise a new
            `MeshTextureVertexColor` on the requested device.
        """

        assert device is None or isinstance(device, (str, torch.device)), (
            "Expected `device` to be `None`, a `str`, or a `torch.device`. "
            f"{type(device)=}"
        )
        assert convention is None, (
            "Expected `convention` to be `None` for a vertex-color texture; "
            "vertex color carries no UV-origin convention. "
            f"{convention=}"
        )

        target_device = self.device if device is None else torch.device(device)
        if self.device == target_device:
            return self
        return MeshTextureVertexColor(
            vertex_color=self.vertex_color.to(device=target_device)
        )
