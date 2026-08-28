from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from data.structures.three_d.camera.extrinsics.conventions import (
    transform_extr_convention,
)
from data.structures.three_d.camera.extrinsics.validation import (
    validate_camera_extrinsics_attributes,
    validate_extr_convention,
    validate_rotation_matrix,
)

_ORTHOGONALITY_REPAIR_ATOL = 1.0e-05


class CameraExtrinsics:
    """A camera's pose: a 4x4 cam2world matrix plus the pose frame it is expressed in."""

    def __init__(
        self,
        extrinsics: Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]],
        extr_convention: str,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Construct a CameraExtrinsics from a 4x4 cam2world matrix and its pose frame.

        Args:
            extrinsics: 4x4 camera-to-world extrinsics matrix as a numpy array, torch.Tensor, or nested numeric list.
            extr_convention: Coordinate-frame convention string.
            device: Optional target device for the extrinsics tensor.
            dtype: Optional target floating dtype for the extrinsics tensor.

        Returns:
            None.
        """
        validate_camera_extrinsics_attributes(
            extrinsics=extrinsics,
            extr_convention=extr_convention,
            device=device,
            dtype=dtype,
        )

        def _normalize_inputs(
            extrinsics: Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]],
            device: Optional[Union[str, torch.device]],
            dtype: Optional[torch.dtype],
        ) -> Tuple[torch.Tensor, torch.device, torch.dtype]:
            if device is not None:
                device = torch.device(device)
            elif isinstance(extrinsics, torch.Tensor):
                device = extrinsics.device
            else:
                device = torch.device("cpu")
            if dtype is not None:
                dtype = dtype
            elif isinstance(extrinsics, torch.Tensor):
                dtype = extrinsics.dtype
            elif isinstance(extrinsics, np.ndarray):
                dtype = torch.as_tensor(extrinsics).dtype
            else:
                dtype = torch.float32
            extrinsics = torch.as_tensor(extrinsics).to(device=device, dtype=dtype)
            return extrinsics, device, dtype

        extrinsics, device, dtype = _normalize_inputs(
            extrinsics=extrinsics,
            device=device,
            dtype=dtype,
        )
        self._extrinsics: torch.Tensor = extrinsics
        self._extr_convention: str = extr_convention
        self._device: torch.device = device
        self._dtype: torch.dtype = dtype

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
    def extr_convention(self) -> str:
        """The pose frame this cam2world matrix is expressed in.

        Args:
            None.

        Returns:
            The pose-frame convention string (standard / opengl / opencv / pytorch3d / arkit).
        """
        return self._extr_convention

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
    def dtype(self) -> torch.dtype:
        """The dtype of the extrinsics tensor.

        Args:
            None.

        Returns:
            The torch dtype of the camera-to-world matrix.
        """
        return self._dtype

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
        """The extr_convention-dispatched physical right axis.

        Args:
            None.

        Returns:
            The unit right-axis length-3 torch.Tensor.
        """
        if self._extr_convention == "standard":
            vec = self._extrinsics[:3, 0]
        elif self._extr_convention == "opengl":
            vec = self._extrinsics[:3, 0]
        elif self._extr_convention == "opencv":
            vec = self._extrinsics[:3, 0]
        elif self._extr_convention == "pytorch3d":
            vec = -self._extrinsics[:3, 0]
        elif self._extr_convention == "arkit":
            vec = -self._extrinsics[:3, 1]
        else:
            assert False, f"Unsupported extr_convention: {self._extr_convention}"
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
        """The extr_convention-dispatched physical forward axis.

        Args:
            None.

        Returns:
            The unit forward-axis length-3 torch.Tensor.
        """
        if self._extr_convention == "standard":
            vec = self._extrinsics[:3, 1]
        elif self._extr_convention == "opengl":
            vec = -self._extrinsics[:3, 2]
        elif self._extr_convention == "opencv":
            vec = self._extrinsics[:3, 2]
        elif self._extr_convention == "pytorch3d":
            vec = self._extrinsics[:3, 2]
        elif self._extr_convention == "arkit":
            vec = self._extrinsics[:3, 2]
        else:
            assert False, f"Unsupported extr_convention: {self._extr_convention}"
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
        """The extr_convention-dispatched physical up axis.

        Args:
            None.

        Returns:
            The unit up-axis length-3 torch.Tensor.
        """
        if self._extr_convention == "standard":
            vec = self._extrinsics[:3, 2]
        elif self._extr_convention == "opengl":
            vec = self._extrinsics[:3, 1]
        elif self._extr_convention == "opencv":
            vec = -self._extrinsics[:3, 1]
        elif self._extr_convention == "pytorch3d":
            vec = self._extrinsics[:3, 1]
        elif self._extr_convention == "arkit":
            vec = -self._extrinsics[:3, 0]
        else:
            assert False, f"Unsupported extr_convention: {self._extr_convention}"
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
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
        extr_convention: Optional[str] = None,
    ) -> "CameraExtrinsics":
        """Return this CameraExtrinsics with tensor placement and pose-frame changes.

        Args:
            device: Target device; ``None`` keeps the current device.
            dtype: Target floating dtype; ``None`` keeps the current dtype.
            non_blocking: Whether tensor moves may be asynchronous.
            copy: Whether tensor moves must allocate new storage even when unchanged.
            extr_convention: Target pose frame; ``None`` keeps the current one.

        Returns:
            This CameraExtrinsics when unchanged, else a new one.
        """

        def _validate_inputs() -> None:
            assert device is None or isinstance(device, (str, torch.device)), (
                "Expected target device to be None, a string, or torch.device. "
                f"{device=}"
            )
            assert dtype is None or isinstance(dtype, torch.dtype), (
                "Expected target dtype to be None or a torch dtype. " f"{dtype=}"
            )
            if dtype is not None:
                assert torch.empty((), dtype=dtype).is_floating_point(), (
                    "Expected target dtype to be floating. " f"{dtype=}"
                )
            assert isinstance(non_blocking, bool), (
                "Expected non_blocking to be a bool. " f"{type(non_blocking)=}"
            )
            assert isinstance(copy, bool), (
                "Expected copy to be a bool. " f"{type(copy)=}"
            )
            assert extr_convention is None or isinstance(extr_convention, str), (
                "Expected target pose frame to be None or a string. "
                f"{extr_convention=}"
            )
            if extr_convention is not None:
                validate_extr_convention(extr_convention)

        _validate_inputs()

        def _normalize_inputs(
            device: Optional[Union[str, torch.device]],
            dtype: Optional[torch.dtype],
            extr_convention: Optional[str],
        ) -> Tuple[torch.device, torch.dtype, str]:
            device = torch.device(device) if device is not None else self._device
            dtype = dtype if dtype is not None else self._dtype
            extr_convention = (
                extr_convention
                if extr_convention is not None
                else self._extr_convention
            )
            return device, dtype, extr_convention

        device, dtype, extr_convention = _normalize_inputs(
            device=device,
            dtype=dtype,
            extr_convention=extr_convention,
        )

        if (
            device == self._device
            and dtype == self._dtype
            and extr_convention == self._extr_convention
            and copy is False
        ):
            return self

        if extr_convention != self._extr_convention:
            extrinsics = transform_extr_convention(
                camera_extrinsics=self,
                target_extr_convention=extr_convention,
            )
        else:
            extrinsics = self._extrinsics

        return CameraExtrinsics(
            extrinsics=extrinsics.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
                copy=copy,
            ),
            extr_convention=extr_convention,
        )

    def transform_extrinsics(
        self,
        scale: Union[int, float, np.ndarray, torch.Tensor],
        rotation: Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]],
        translation: Union[
            np.ndarray,
            torch.Tensor,
            Tuple[Union[int, float], Union[int, float], Union[int, float]],
            List[Union[int, float]],
        ],
    ) -> "CameraExtrinsics":
        """Return this CameraExtrinsics under a similarity transform of its pose.

        Args:
            scale: Scalar similarity scale factor as a number, numpy array, or torch.Tensor.
            rotation: 3x3 rotation matrix as a numpy array, torch.Tensor, or nested numeric list.
            translation: Length-3 translation as a numpy array, torch.Tensor, tuple, or list.

        Returns:
            A new CameraExtrinsics with the transformed cam2world pose.
        """

        def _validate_inputs() -> None:
            assert isinstance(scale, (int, float, np.ndarray, torch.Tensor)), (
                "Expected transform scale to be a number, numpy array, or torch.Tensor. "
                f"{type(scale)=}"
            )
            if isinstance(scale, np.ndarray):
                assert scale.size == 1, (
                    "Expected transform scale array to contain one value. "
                    f"{scale.shape=}"
                )
                assert np.issubdtype(scale.dtype, np.number), (
                    "Expected transform scale array to be numeric. " f"{scale.dtype=}"
                )
            if isinstance(scale, torch.Tensor):
                assert scale.numel() == 1, (
                    "Expected transform scale tensor to contain one value. "
                    f"{scale.shape=}"
                )
                assert scale.is_floating_point(), (
                    "Expected transform scale tensor to be floating. " f"{scale.dtype=}"
                )
            validate_rotation_matrix(rotation)
            assert isinstance(translation, (np.ndarray, torch.Tensor, tuple, list)), (
                "Expected transform translation to be a numpy array, torch.Tensor, "
                "tuple, or list. "
                f"{type(translation)=}"
            )
            if isinstance(translation, np.ndarray):
                assert translation.shape == (3,), (
                    "Expected transform translation shape to be length 3. "
                    f"{translation.shape=}"
                )
                assert np.issubdtype(translation.dtype, np.number), (
                    "Expected transform translation array to be numeric. "
                    f"{translation.dtype=}"
                )
            if isinstance(translation, torch.Tensor):
                assert translation.shape == (3,), (
                    "Expected transform translation shape to be length 3. "
                    f"{translation.shape=}"
                )
                assert translation.is_floating_point(), (
                    "Expected transform translation tensor to be floating. "
                    f"{translation.dtype=}"
                )
            if isinstance(translation, (tuple, list)):
                assert len(translation) == 3, (
                    "Expected transform translation to have length 3. "
                    f"{translation=}"
                )
                assert all(isinstance(value, (int, float)) for value in translation), (
                    "Expected transform translation values to be numeric. "
                    f"{translation=}"
                )

        _validate_inputs()

        def _normalize_inputs(
            scale: Union[int, float, np.ndarray, torch.Tensor],
            rotation: Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]],
            translation: Union[
                np.ndarray,
                torch.Tensor,
                Tuple[Union[int, float], Union[int, float], Union[int, float]],
                List[Union[int, float]],
            ],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if isinstance(scale, torch.Tensor):
                scale = scale.to(device=self._device, dtype=self._dtype)
            else:
                scale = torch.as_tensor(scale, device=self._device, dtype=self._dtype)
            scale = scale.reshape(())
            if isinstance(rotation, torch.Tensor):
                rotation = rotation.to(device=self._device, dtype=self._dtype)
            else:
                rotation = torch.as_tensor(
                    rotation,
                    device=self._device,
                    dtype=self._dtype,
                )
            if isinstance(translation, torch.Tensor):
                translation = translation.to(device=self._device, dtype=self._dtype)
            else:
                translation = torch.as_tensor(
                    translation,
                    device=self._device,
                    dtype=self._dtype,
                )
            translation = translation.reshape(3)
            return scale, rotation, translation

        scale, rotation, translation = _normalize_inputs(
            scale=scale,
            rotation=rotation,
            translation=translation,
        )

        rotation_c2w = self._extrinsics[:3, :3]
        translation_c2w = self._extrinsics[:3, 3]
        rotation_c2w_new = rotation @ rotation_c2w
        translation_c2w_new = scale * (rotation @ translation_c2w) + translation

        extrinsics_new = torch.eye(
            4,
            dtype=self._dtype,
            device=self._device,
        )
        extrinsics_new[:3, :3] = rotation_c2w_new
        extrinsics_new[:3, 3] = translation_c2w_new
        extrinsics_new[:3, :3] = _stabilize_rotation_matrix(extrinsics_new[:3, :3])

        return CameraExtrinsics(
            extrinsics=extrinsics_new,
            extr_convention=self._extr_convention,
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
