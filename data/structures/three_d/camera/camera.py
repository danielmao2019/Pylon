from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
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
        device: Union[str, torch.device] = torch.device("cuda"),
    ) -> None:
        """Construct a Camera from a CameraIntrinsics and a CameraExtrinsics.

        Args:
            intrinsics: The camera's CameraIntrinsics ("what the camera is").
            extrinsics: The camera's CameraExtrinsics ("where the camera is").
            name: Optional camera name.
            id: Optional camera id.
            device: Device the camera tensors live on, a string or torch.device.

        Returns:
            None.
        """
        validate_camera_attributes(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            name=name,
            id=id,
            device=device,
        )
        self._device: torch.device = torch.device(device)
        self._intrinsics: CameraIntrinsics = intrinsics.to(device=self._device)
        self._extrinsics: CameraExtrinsics = extrinsics.to(device=self._device)
        self._name: Optional[str] = name
        self._id: Optional[int] = id

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

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        convention: Optional[str] = None,
    ) -> "Camera":
        """Return this Camera on a target device / extrinsics convention.

        Args:
            device: Target device; ``None`` keeps the current device.
            convention: Target extrinsics convention; ``None`` keeps the current one.

        Returns:
            This Camera when unchanged, else a new one.
        """
        # Input validations
        assert device is None or isinstance(device, (str, torch.device)), (
            "Expected target device to be None, a string, or torch.device. "
            f"{device=}"
        )
        assert convention is None or isinstance(convention, str), (
            "Expected target convention to be None or a string. " f"{convention=}"
        )

        # Input normalizations
        target_device = torch.device(device) if device is not None else self._device
        if target_device == self._device and (
            convention is None or convention == self._extrinsics.convention
        ):
            return self

        intrinsics = self._intrinsics.to(device=target_device)
        extrinsics = self._extrinsics.to(device=target_device, convention=convention)
        return Camera(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            name=self._name,
            id=self._id,
            device=target_device,
        )

    def scale_intrinsics(
        self,
        resolution: Optional[Tuple[int, int]] = None,
        scale: Optional[
            Union[Union[int, float], Tuple[Union[int, float], Union[int, float]]]
        ] = None,
    ) -> "Camera":
        """Return this Camera with its CameraIntrinsics scaled.

        Args:
            resolution: Optional target image resolution as ``(height, width)``.
            scale: Optional uniform scale, or a per-axis ``(sx, sy)`` tuple.

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
        )

    def transform(
        self,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> "Camera":
        """Return this Camera under a similarity transform of its CameraExtrinsics.

        Args:
            scale: Similarity scale factor.
            rotation: 3x3 rotation matrix as a float32 numpy array.
            translation: Length-3 translation as a float32 numpy array.

        Returns:
            A new Camera with the transformed CameraExtrinsics pose.
        """
        extrinsics = self._extrinsics.transform(
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
