from typing import Optional, Union

import numpy as np
import torch

from data.structures.three_d.camera.extrinsics.conventions import transform_convention
from data.structures.three_d.camera.extrinsics.validation import (
    validate_camera_convention,
    validate_camera_extrinsics_attributes,
    validate_rotation_matrix,
)

_ORTHOGONALITY_REPAIR_ATOL = 1.0e-05


class CameraExtrinsics:
    """A camera's pose: a 4x4 cam2world matrix plus its coordinate-frame convention."""

    def __init__(
        self,
        extrinsics: torch.Tensor,
        convention: str,
        device: Union[str, torch.device] = torch.device("cuda"),
    ) -> None:
        """Construct a CameraExtrinsics from a 4x4 cam2world matrix and a convention.

        Args:
            extrinsics: 4x4 camera-to-world extrinsics matrix as a torch.Tensor.
            convention: Coordinate-frame convention string.
            device: Device the extrinsics live on, a string or torch.device.

        Returns:
            None.
        """
        validate_camera_extrinsics_attributes(
            extrinsics=extrinsics,
            convention=convention,
            device=device,
        )
        self._device: torch.device = torch.device(device)
        self._extrinsics: torch.Tensor = extrinsics.to(device=self._device)
        self._convention: str = convention

    @property
    def extrinsics(self) -> torch.Tensor:
        """The 4x4 camera-to-world extrinsics matrix.

        Args:
            None.

        Returns:
            The 4x4 camera-to-world extrinsics torch.Tensor.
        """
        return self._extrinsics

    @property
    def convention(self) -> str:
        """The coordinate-frame convention.

        Args:
            None.

        Returns:
            The convention string (standard / opengl / opencv / pytorch3d / arkit).
        """
        return self._convention

    @property
    def device(self) -> torch.device:
        """The device the extrinsics live on.

        Args:
            None.

        Returns:
            The device.
        """
        return self._device

    @property
    def w2c(self) -> torch.Tensor:
        """The world-to-camera matrix (inverse of extrinsics).

        Args:
            None.

        Returns:
            The 4x4 world-to-camera torch.Tensor.
        """
        return torch.inverse(self._extrinsics)

    @property
    def center(self) -> torch.Tensor:
        """The camera center.

        Args:
            None.

        Returns:
            The camera center ``extrinsics[:3, 3]`` as a length-3 torch.Tensor.
        """
        center = self._extrinsics[:3, 3]
        assert center.shape == (3,), f"{center.shape=}"
        return center

    @property
    def right(self) -> torch.Tensor:
        """The convention-dispatched physical right axis.

        Args:
            None.

        Returns:
            The unit right-axis length-3 torch.Tensor.
        """
        if self._convention == "standard":
            vec = self._extrinsics[:3, 0]
        elif self._convention == "opengl":
            vec = self._extrinsics[:3, 0]
        elif self._convention == "opencv":
            vec = self._extrinsics[:3, 0]
        elif self._convention == "pytorch3d":
            vec = -self._extrinsics[:3, 0]
        elif self._convention == "arkit":
            vec = -self._extrinsics[:3, 1]
        else:
            assert False, f"Unsupported convention: {self._convention}"
        norm = torch.norm(vec)
        assert torch.isclose(
            norm,
            torch.tensor(1.0, dtype=vec.dtype, device=vec.device),
            atol=1.0e-05,
            rtol=0.0,
        ), f"Right vector must be unit, got norm {float(norm)}"
        return vec

    @property
    def forward(self) -> torch.Tensor:
        """The convention-dispatched physical forward axis.

        Args:
            None.

        Returns:
            The unit forward-axis length-3 torch.Tensor.
        """
        if self._convention == "standard":
            vec = self._extrinsics[:3, 1]
        elif self._convention == "opengl":
            vec = -self._extrinsics[:3, 2]
        elif self._convention == "opencv":
            vec = self._extrinsics[:3, 2]
        elif self._convention == "pytorch3d":
            vec = self._extrinsics[:3, 2]
        elif self._convention == "arkit":
            vec = self._extrinsics[:3, 2]
        else:
            assert False, f"Unsupported convention: {self._convention}"
        norm = torch.norm(vec)
        assert torch.isclose(
            norm,
            torch.tensor(1.0, dtype=vec.dtype, device=vec.device),
            atol=1.0e-05,
            rtol=0.0,
        ), f"Forward vector must be unit, got norm {float(norm)}"
        return vec

    @property
    def up(self) -> torch.Tensor:
        """The convention-dispatched physical up axis.

        Args:
            None.

        Returns:
            The unit up-axis length-3 torch.Tensor.
        """
        if self._convention == "standard":
            vec = self._extrinsics[:3, 2]
        elif self._convention == "opengl":
            vec = self._extrinsics[:3, 1]
        elif self._convention == "opencv":
            vec = -self._extrinsics[:3, 1]
        elif self._convention == "pytorch3d":
            vec = self._extrinsics[:3, 1]
        elif self._convention == "arkit":
            vec = -self._extrinsics[:3, 0]
        else:
            assert False, f"Unsupported convention: {self._convention}"
        norm = torch.norm(vec)
        assert torch.isclose(
            norm,
            torch.tensor(1.0, dtype=vec.dtype, device=vec.device),
            atol=1.0e-05,
            rtol=0.0,
        ), f"Up vector must be unit, got norm {float(norm)}"
        return vec

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        convention: Optional[str] = None,
    ) -> "CameraExtrinsics":
        """Return this CameraExtrinsics on a target device / convention.

        Args:
            device: Target device; ``None`` keeps the current device.
            convention: Target convention; ``None`` keeps the current convention.

        Returns:
            This CameraExtrinsics when unchanged, else a new one.
        """
        # Input validations
        assert device is None or isinstance(device, (str, torch.device)), (
            "Expected target device to be None, a string, or torch.device. "
            f"{device=}"
        )
        assert convention is None or isinstance(convention, str), (
            "Expected target convention to be None or a string. " f"{convention=}"
        )
        if convention is not None:
            validate_camera_convention(convention)

        # Input normalizations
        target_device = torch.device(device) if device is not None else self._device
        target_convention = convention if convention is not None else self._convention

        if target_device == self._device and target_convention == self._convention:
            return self

        if target_convention != self._convention:
            extrinsics = transform_convention(
                camera_extrinsics=self,
                target_convention=target_convention,
            )
        else:
            extrinsics = self._extrinsics

        return CameraExtrinsics(
            extrinsics=extrinsics.to(device=target_device),
            convention=target_convention,
            device=target_device,
        )

    def transform(
        self,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> "CameraExtrinsics":
        """Return this CameraExtrinsics under a similarity transform of its pose.

        Args:
            scale: Similarity scale factor.
            rotation: 3x3 rotation matrix as a float32 numpy array.
            translation: Length-3 translation as a float32 numpy array.

        Returns:
            A new CameraExtrinsics with the transformed cam2world pose.
        """
        # Input validations
        assert isinstance(scale, (int, float)), (
            "Expected transform scale to be a number. " f"{type(scale)=}"
        )
        assert isinstance(rotation, np.ndarray), (
            "Expected transform rotation to be a numpy array. " f"{type(rotation)=}"
        )
        assert rotation.shape == (3, 3), (
            "Expected transform rotation shape to be 3x3. " f"{rotation.shape=}"
        )
        assert rotation.dtype == np.float32, (
            "Expected transform rotation dtype to be float32. " f"{rotation.dtype=}"
        )
        validate_rotation_matrix(rotation)
        assert isinstance(translation, np.ndarray), (
            "Expected transform translation to be a numpy array. "
            f"{type(translation)=}"
        )
        assert translation.shape == (3,), (
            "Expected transform translation shape to be length 3. "
            f"{translation.shape=}"
        )
        assert translation.dtype == np.float32, (
            "Expected transform translation dtype to be float32. "
            f"{translation.dtype=}"
        )

        # Input normalizations
        rotation_tensor = torch.from_numpy(rotation).to(
            device=self._device,
            dtype=self._extrinsics.dtype,
        )
        translation_tensor = torch.from_numpy(translation).to(
            device=self._device,
            dtype=self._extrinsics.dtype,
        )

        rotation_c2w = self._extrinsics[:3, :3]
        translation_c2w = self._extrinsics[:3, 3]
        rotation_c2w_new = rotation_tensor @ rotation_c2w
        translation_c2w_new = scale * (rotation_tensor @ translation_c2w) + (
            translation_tensor
        )

        extrinsics_new = torch.eye(
            4,
            dtype=self._extrinsics.dtype,
            device=self._extrinsics.device,
        )
        extrinsics_new[:3, :3] = rotation_c2w_new
        extrinsics_new[:3, 3] = translation_c2w_new
        extrinsics_new[:3, :3] = _stabilize_rotation_matrix(extrinsics_new[:3, :3])

        return CameraExtrinsics(
            extrinsics=extrinsics_new,
            convention=self._convention,
            device=self._device,
        )


def _stabilize_rotation_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Project a near-orthogonal (3, 3) rotation onto the nearest proper rotation.

    Args:
        rotation: A near-orthogonal 3x3 rotation as a float32 or float64 torch.Tensor.

    Returns:
        The nearest proper rotation matrix, in the received dtype.
    """
    # Input validations
    assert isinstance(rotation, torch.Tensor), (
        "Expected rotation matrix to be a torch.Tensor. " f"{type(rotation)=}"
    )
    assert rotation.shape == (3, 3), (
        "Expected rotation matrix shape to be 3x3. " f"{rotation.shape=}"
    )
    assert rotation.dtype in (torch.float32, torch.float64), (
        "Expected rotation matrix dtype to be float32 or float64. " f"{rotation.dtype=}"
    )

    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    should_be_identity = rotation @ rotation.transpose(-1, -2)
    orthogonality_residual = float(torch.max(torch.abs(should_be_identity - identity)))
    determinant_residual = abs(float(torch.linalg.det(rotation)) - 1.0)
    assert (
        max(orthogonality_residual, determinant_residual) <= _ORTHOGONALITY_REPAIR_ATOL
    ), (
        "Expected near-orthogonal rotation matrix before stabilization. "
        f"{orthogonality_residual=} {determinant_residual=} {_ORTHOGONALITY_REPAIR_ATOL=}"
    )

    u, _, v_h = torch.linalg.svd(rotation)
    rotation_fixed = u @ v_h
    if float(torch.linalg.det(rotation_fixed)) < 0.0:
        u[:, -1] = -u[:, -1]
        rotation_fixed = u @ v_h
    validate_rotation_matrix(rotation_fixed)
    return rotation_fixed
