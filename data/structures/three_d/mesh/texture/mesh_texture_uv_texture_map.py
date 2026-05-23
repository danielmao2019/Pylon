from typing import Optional, Tuple, Union

import torch

from data.structures.three_d.mesh.texture.conventions import (
    transform_verts_uvs_convention,
)
from data.structures.three_d.mesh.texture.mesh_texture import MeshTexture
from data.structures.three_d.mesh.texture.validate_uv_texture_map import (
    validate_uv_texture_map,
)


class MeshTextureUVTextureMap(MeshTexture):
    """UV-atlas texture.

    Holds a UV texture image `uv_texture_map` `[H, W, 3]`, a UV-coordinate table
    `verts_uvs` `[U, 2]`, face-to-UV indices `faces_uvs` `[F, 3]`, and the
    UV-origin convention of `verts_uvs`.

    Args:
        uv_texture_map: UV texture map in CHW, HWC, NCHW, or NHWC layout with
            uint8 `[0, 255]` or float32 `[0, 1]` values.
        verts_uvs: UV-coordinate table `[U, 2]`.
        faces_uvs: Face-to-UV index tensor `[F, 3]`.
        convention: UV-origin convention for `verts_uvs`. `obj` means `v=0` is
            the bottom edge. `top_left` means `v=0` is the top edge.

    Returns:
        None.
    """

    uv_texture_map: torch.Tensor
    verts_uvs: torch.Tensor
    faces_uvs: torch.Tensor
    convention: str

    def __init__(
        self,
        uv_texture_map: torch.Tensor,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        convention: str,
    ) -> None:
        """Initialize one UV-atlas texture.

        Args:
            uv_texture_map: UV texture map in CHW, HWC, NCHW, or NHWC layout
                with uint8 `[0, 255]` or float32 `[0, 1]` values.
            verts_uvs: UV-coordinate table `[U, 2]`.
            faces_uvs: Face-to-UV index tensor `[F, 3]`.
            convention: UV-origin convention for `verts_uvs`.

        Returns:
            None.
        """

        def _validate_inputs() -> None:
            validate_uv_texture_map(
                uv_texture_map=uv_texture_map,
                verts_uvs=verts_uvs,
                faces_uvs=faces_uvs,
                convention=convention,
            )

        _validate_inputs()

        def _normalize_inputs() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                MeshTextureUVTextureMap.normalize_uv_texture_map(
                    uv_texture_map=uv_texture_map
                ),
                verts_uvs.contiguous(),
                faces_uvs.to(dtype=torch.int64).contiguous(),
            )

        uv_texture_map, verts_uvs, faces_uvs = _normalize_inputs()

        self.uv_texture_map = uv_texture_map
        self.verts_uvs = verts_uvs
        self.faces_uvs = faces_uvs
        self.convention = convention

    @staticmethod
    def normalize_uv_texture_map(uv_texture_map: torch.Tensor) -> torch.Tensor:
        """Normalize one UV texture map to the canonical format.

        Args:
            uv_texture_map: UV texture map in CHW, HWC, NCHW, or NHWC layout
                with uint8 `[0, 255]` or float32 `[0, 1]` values.

        Returns:
            UV texture map in contiguous float32 HWC layout with values in
            `[0, 1]`.
        """

        if uv_texture_map.ndim == 4:
            uv_texture_map = uv_texture_map[0]
        if uv_texture_map.shape[0] == 3:
            uv_texture_map = uv_texture_map.permute(1, 2, 0)
        if uv_texture_map.dtype == torch.uint8:
            return uv_texture_map.to(dtype=torch.float32).div(255.0).contiguous()
        return uv_texture_map.contiguous()

    @property
    def device(self) -> torch.device:
        """Return the device the UV-texture tensors live on.

        Args:
            None.

        Returns:
            The `torch.device` the UV-texture tensors live on.
        """

        return self.uv_texture_map.device

    def to(
        self,
        device: Union[str, torch.device, None] = None,
        convention: Optional[str] = None,
    ) -> "MeshTextureUVTextureMap":
        """Return this texture on a target device and/or UV-origin convention.

        Args:
            device: Optional target device.
            convention: Optional target UV-origin convention.

        Returns:
            This texture when both the device and convention already match,
            otherwise a new `MeshTextureUVTextureMap` on the requested target.
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

        _validate_inputs()

        target_device = self.device if device is None else torch.device(device)
        target_convention = self.convention if convention is None else convention
        if self.device == target_device and self.convention == target_convention:
            return self

        target_verts_uvs = self.verts_uvs
        if target_convention != self.convention:
            target_verts_uvs = transform_verts_uvs_convention(
                verts_uvs=self.verts_uvs,
                source_convention=self.convention,
                target_convention=target_convention,
            )

        return MeshTextureUVTextureMap(
            uv_texture_map=self.uv_texture_map.to(device=target_device),
            verts_uvs=target_verts_uvs.to(device=target_device),
            faces_uvs=self.faces_uvs.to(device=target_device),
            convention=target_convention,
        )
