from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.extrinsics.validation import (
    validate_extr_convention,
)
from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
from data.structures.three_d.camera.intrinsics.validation import (
    validate_intr_convention,
)
from data.structures.three_d.camera.io import (
    deserialize_cameras,
    load_cameras,
    save_cameras,
    serialize_cameras,
)
from data.structures.three_d.camera.validation import validate_camera_attributes


class Camera:
    """A camera: a CameraIntrinsics and a CameraExtrinsics, plus name / id / device."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        name: Optional[str] = None,
        id: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Construct a Camera from a CameraIntrinsics and a CameraExtrinsics.

        Args:
            intrinsics: The camera's CameraIntrinsics ("what the camera is").
            extrinsics: The camera's CameraExtrinsics ("where the camera is").
            name: Optional camera name.
            id: Optional camera id.
            device: Optional target device for the camera tensors.
            dtype: Optional target floating dtype for the camera tensors.

        Returns:
            None.
        """
        validate_camera_attributes(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            name=name,
            id=id,
            device=device,
            dtype=dtype,
        )
        if device is not None or dtype is not None:
            intrinsics = intrinsics.to(device=device, dtype=dtype)
            extrinsics = extrinsics.to(device=device, dtype=dtype)
        assert intrinsics.device == extrinsics.device, (
            "Expected Camera components to share device. "
            f"{intrinsics.device=} {extrinsics.device=}"
        )
        assert intrinsics.dtype == extrinsics.dtype, (
            "Expected Camera components to share dtype. "
            f"{intrinsics.dtype=} {extrinsics.dtype=}"
        )
        self._intrinsics: CameraIntrinsics = intrinsics
        self._extrinsics: CameraExtrinsics = extrinsics
        self._name: Optional[str] = name
        self._id: Optional[int] = id
        self._device: torch.device = intrinsics.device
        self._dtype: torch.dtype = intrinsics.dtype

    @property
    def intrinsics(self) -> CameraIntrinsics:
        """The camera's CameraIntrinsics ("what the camera is").

        Args:
            None.

        Returns:
            The CameraIntrinsics.
        """
        return self._intrinsics

    @property
    def extrinsics(self) -> CameraExtrinsics:
        """The camera's CameraExtrinsics ("where the camera is").

        Args:
            None.

        Returns:
            The CameraExtrinsics.
        """
        return self._extrinsics

    @property
    def name(self) -> Optional[str]:
        """The camera name.

        Args:
            None.

        Returns:
            The camera name or None.
        """
        return self._name

    @property
    def id(self) -> Optional[int]:
        """The camera id.

        Args:
            None.

        Returns:
            The camera id or None.
        """
        return self._id

    @property
    def device(self) -> torch.device:
        """The device the camera tensors live on.

        Args:
            None.

        Returns:
            The device.
        """
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """The dtype shared by the camera tensors.

        Args:
            None.

        Returns:
            The torch dtype shared by intrinsics and extrinsics.
        """
        return self._dtype

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
        intr_convention: Optional[str] = None,
        extr_convention: Optional[str] = None,
    ) -> "Camera":
        """Return this Camera with tensor placement and convention changes.

        Each half is named for the half it converts, because neither is the one a
        bare convention would mean.

        Args:
            device: Target device; ``None`` keeps the current device.
            dtype: Target floating dtype; ``None`` keeps the current dtype.
            non_blocking: Whether tensor moves may be asynchronous.
            copy: Whether tensor moves must allocate new storage even when unchanged.
            intr_convention: Target image-plane frame; ``None`` keeps the current one.
            extr_convention: Target pose frame; ``None`` keeps the current one.

        Returns:
            This Camera when unchanged, else a new one.
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
            assert intr_convention is None or isinstance(intr_convention, str), (
                "Expected target image-plane frame to be None or a string. "
                f"{intr_convention=}"
            )
            if intr_convention is not None:
                validate_intr_convention(intr_convention=intr_convention)
            assert extr_convention is None or isinstance(extr_convention, str), (
                "Expected target pose frame to be None or a string. "
                f"{extr_convention=}"
            )
            if extr_convention is not None:
                validate_extr_convention(extr_convention=extr_convention)

        _validate_inputs()

        def _normalize_inputs(
            device: Optional[Union[str, torch.device]],
            dtype: Optional[torch.dtype],
            intr_convention: Optional[str],
            extr_convention: Optional[str],
        ) -> Tuple[torch.device, torch.dtype, str, str]:
            device = torch.device(device) if device is not None else self._device
            dtype = dtype if dtype is not None else self._dtype
            intr_convention = (
                intr_convention
                if intr_convention is not None
                else self._intrinsics.intr_convention
            )
            extr_convention = (
                extr_convention
                if extr_convention is not None
                else self._extrinsics.extr_convention
            )
            return device, dtype, intr_convention, extr_convention

        device, dtype, intr_convention, extr_convention = _normalize_inputs(
            device=device,
            dtype=dtype,
            intr_convention=intr_convention,
            extr_convention=extr_convention,
        )

        if (
            device == self._device
            and dtype == self._dtype
            and intr_convention == self._intrinsics.intr_convention
            and extr_convention == self._extrinsics.extr_convention
            and copy is False
        ):
            return self

        intrinsics = self._intrinsics.to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
            copy=copy,
            intr_convention=intr_convention,
        )
        extrinsics = self._extrinsics.to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
            copy=copy,
            extr_convention=extr_convention,
        )
        return Camera(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            name=self._name,
            id=self._id,
            device=device,
            dtype=dtype,
        )

    def transform_intrinsics(
        self,
        transform: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> "Camera":
        """Return this Camera with its CameraIntrinsics restated onto another image.

        The affine is paired with that image's own raster, because a 3x3 carries no
        size of its own.

        Args:
            transform: Pixel-frame affine as a ``(3, 3)`` float32 torch.Tensor whose last row is ``[0, 0, 1]``.
            resolution: The target image's own resolution as ``(height, width)``.

        Returns:
            A new Camera whose CameraIntrinsics is stated against ``resolution``.
        """
        intrinsics = self._intrinsics.transform_intrinsics(
            transform=transform,
            resolution=resolution,
        )
        return Camera(
            intrinsics=intrinsics,
            extrinsics=self._extrinsics,
            name=self._name,
            id=self._id,
            device=self._device,
            dtype=self._dtype,
        )

    def scale_intrinsics(
        self,
        resolution: Optional[
            Union[int, Tuple[int, int], List[int], np.ndarray, torch.Tensor]
        ] = None,
        scale: Optional[
            Union[
                int,
                float,
                Tuple[
                    Union[int, float, torch.Tensor],
                    Union[int, float, torch.Tensor],
                ],
                List[Union[int, float, torch.Tensor]],
                np.ndarray,
                torch.Tensor,
            ]
        ] = None,
    ) -> "Camera":
        """Return this Camera with its CameraIntrinsics scaled.

        Args:
            resolution: Optional target image resolution as one integer side or ``(height, width)``.
            scale: Optional uniform scale, or a per-axis ``(sx, sy)`` pair.

        Returns:
            A new Camera with scaled CameraIntrinsics.
        """
        intrinsics = self._intrinsics.scale_intrinsics(
            resolution=resolution,
            scale=scale,
        )
        return Camera(
            intrinsics=intrinsics,
            extrinsics=self._extrinsics,
            name=self._name,
            id=self._id,
            device=self._device,
            dtype=self._dtype,
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
    ) -> "Camera":
        """Return this Camera under a similarity transform of its CameraExtrinsics.

        Args:
            scale: Scalar similarity scale factor as a number, numpy array, or torch.Tensor.
            rotation: 3x3 rotation matrix as a numpy array, torch.Tensor, or nested numeric list.
            translation: Length-3 translation as a numpy array, torch.Tensor, tuple, or list.

        Returns:
            A new Camera with the transformed CameraExtrinsics pose.
        """
        extrinsics = self._extrinsics.transform_extrinsics(
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        return Camera(
            intrinsics=self._intrinsics,
            extrinsics=extrinsics,
            name=self._name,
            id=self._id,
            device=self._device,
            dtype=self._dtype,
        )

    def serialize(self, format: str = "json") -> Dict[str, Any]:
        """Serialize this Camera into a single-form payload.

        Single-camera convenience wrapper over the plural `serialize_cameras`
        dispatcher, which normalizes this single Camera to the single-form payload.

        Args:
            format: Serialization format, either `json` or `npz`.

        Returns:
            Single-form Camera payload for the requested format.
        """
        return serialize_cameras(cameras=self, format=format)

    @classmethod
    def deserialize(
        cls,
        payload: Dict[str, Any],
        device: Optional[Union[str, torch.device]] = None,
        format: str = "json",
    ) -> "Camera":
        """Deserialize one Camera from a single-form payload.

        Single-camera convenience wrapper over the plural `deserialize_cameras`
        dispatcher; asserts the payload was in single form so the result is a
        single Camera.

        Args:
            payload: Single-form Camera payload for the specified format.
            device: Target device for the deserialized Camera.
            format: Serialization format, either `json` or `npz`.

        Returns:
            Camera object represented by the payload.
        """
        camera = deserialize_cameras(
            payload=payload,
            device=device,
            format=format,
        )
        assert isinstance(camera, cls), (
            "Expected Camera.deserialize payload to be a single-form payload "
            f"yielding one Camera. {type(camera)=}"
        )
        return camera

    def save(self, camera_path: Path) -> None:
        """Save this Camera to a `.npz` or `.json` file.

        Single-camera convenience wrapper over the plural `save_cameras`
        dispatcher, which normalizes this single Camera to the single-form file.

        Args:
            camera_path: Output `.npz` or `.json` filepath.

        Returns:
            None.
        """
        save_cameras(cameras=self, cameras_path=camera_path)

    @classmethod
    def load(
        cls,
        camera_path: Path,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "Camera":
        """Load one Camera from a `.npz` or `.json` file.

        Single-camera convenience wrapper over the plural `load_cameras`
        dispatcher; asserts the file held a single form so the result is a single
        Camera.

        Args:
            camera_path: Input `.npz` or `.json` filepath.
            device: Target device for the loaded Camera.

        Returns:
            Camera object loaded from disk.
        """
        camera = load_cameras(cameras_path=camera_path, device=device)
        assert isinstance(camera, cls), (
            "Expected Camera.load file to hold a single-form payload yielding one "
            f"Camera. {type(camera)=}"
        )
        return camera
