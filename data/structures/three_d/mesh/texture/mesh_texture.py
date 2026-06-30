import abc
from typing import Optional, Union

import torch


class MeshTexture(abc.ABC):
    """Abstract base for a mesh's texture.

    Concrete subclasses own the representation-specific tensors and validation.

    Args:
        None.

    Returns:
        None.
    """

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """Return the device the texture's tensors live on.

        Args:
            None.

        Returns:
            The `torch.device` the texture's tensors live on.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def to(
        self,
        device: Union[str, torch.device, None] = None,
        verts_uvs_convention: Optional[str] = None,
    ) -> "MeshTexture":
        """Return this texture on a target device and/or UV-origin convention.

        Args:
            device: Optional target device.
            verts_uvs_convention: Optional target UV-origin convention.

        Returns:
            This texture on the requested device and/or convention.
        """

        raise NotImplementedError
