import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from data.structures.three_d.camera.conventions import transform_convention
from data.structures.three_d.camera.io import (
    deserialize_cameras,
    load_cameras,
    save_cameras,
    serialize_cameras,
)
from data.structures.three_d.camera.scaling import scale_intrinsics
from data.structures.three_d.camera.validation import (
    validate_camera_convention,
    validate_camera_extrinsics,
    validate_camera_intrinsics,
    validate_rotation_matrix,
)

_ORTHOGONALITY_REPAIR_ATOL = 1.0e-05


class Camera:

    def __init__(
        self,
        intrinsics: Optional[torch.Tensor],
        extrinsics: torch.Tensor,
        convention: str,
        name: Optional[str] = None,
        id: Optional[int] = None,
        device: Union[str, torch.device] = torch.device("cuda"),
    ) -> None:
        def _validate_inputs() -> None:
            assert intrinsics is None or isinstance(intrinsics, torch.Tensor), (
                "Expected Camera intrinsics to be None or a torch.Tensor. "
                f"{type(intrinsics)=}"
            )
            if intrinsics is not None:
                validate_camera_intrinsics(intrinsics)

            validate_camera_extrinsics(extrinsics)

            validate_camera_convention(convention)

            assert name is None or isinstance(name, str), (
                "Expected Camera name to be None or a string. " f"{type(name)=}"
            )

            assert id is None or isinstance(id, int), (
                "Expected Camera id to be None or an integer. " f"{type(id)=}"
            )

            assert isinstance(device, (str, torch.device)), (
                "Expected Camera device to be a string or torch.device. "
                f"{type(device)=}"
            )

        _validate_inputs()

        def _normalize_inputs(
            intrinsics: Optional[torch.Tensor],
            extrinsics: torch.Tensor,
            device: Union[str, torch.device],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.device]:
            device = torch.device(device)
            if intrinsics is None:
                intrinsics = torch.eye(3, dtype=torch.float32, device=device)
            else:
                intrinsics = intrinsics.to(device=device)
            extrinsics = extrinsics.to(device=device)
            validate_camera_intrinsics(intrinsics)
            validate_camera_extrinsics(extrinsics)
            return intrinsics, extrinsics, device

        intrinsics, extrinsics, device = _normalize_inputs(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            device=device,
        )

        self._intrinsics = intrinsics
        self._extrinsics = extrinsics
        self._convention = convention
        self._name = name
        self._id = id
        self._device = device

    @property
    def intrinsics(self) -> torch.Tensor:
        return self._intrinsics

    @property
    def extrinsics(self) -> torch.Tensor:
        return self._extrinsics

    @property
    def convention(self) -> str:
        return self._convention

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def id(self) -> Optional[int]:
        return self._id

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def w2c(self) -> torch.Tensor:
        return torch.inverse(self._extrinsics)

    @property
    def fx(self) -> float:
        return float(self._intrinsics[0, 0])

    @property
    def fy(self) -> float:
        return float(self._intrinsics[1, 1])

    @property
    def cx(self) -> float:
        return float(self._intrinsics[0, 2])

    @property
    def cy(self) -> float:
        return float(self._intrinsics[1, 2])

    @property
    def fov(self) -> Tuple[float, float]:
        horizontal_fov = 2.0 * math.atan(self.cx / self.fx) * 180.0 / math.pi
        vertical_fov = 2.0 * math.atan(self.cy / self.fy) * 180.0 / math.pi
        return (horizontal_fov, vertical_fov)

    @property
    def center(self) -> torch.Tensor:
        center = self._extrinsics[:3, 3]
        assert center.shape == (3,), f"{center.shape=}"
        return center

    @property
    def right(self) -> torch.Tensor:
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
    ) -> "Camera":
        def _validate_inputs() -> None:
            assert device is None or isinstance(device, (str, torch.device)), (
                "Expected target device to be None, a string, or torch.device. "
                f"{device=}"
            )

            assert convention is None or isinstance(convention, str), (
                "Expected target Camera convention to be None or a string. "
                f"{convention=}"
            )
            if convention is not None:
                validate_camera_convention(convention)

        _validate_inputs()

        def _normalize_inputs(
            device: Optional[Union[str, torch.device]],
            convention: Optional[str],
        ) -> Tuple[torch.device, str]:
            device = torch.device(device) if device is not None else self._device
            convention = convention if convention is not None else self._convention
            assert isinstance(device, torch.device), (
                "Expected normalized target device to be torch.device. "
                f"{type(device)=}"
            )
            assert isinstance(convention, str), (
                "Expected normalized target convention to be a string. "
                f"{type(convention)=}"
            )
            return device, convention

        device, convention = _normalize_inputs(
            device=device,
            convention=convention,
        )

        if device == self._device and convention == self._convention:
            return self

        extrinsics = (
            transform_convention(
                camera=self,
                target_convention=convention,
            )
            if convention != self._convention
            else self._extrinsics
        )

        return Camera(
            intrinsics=self._intrinsics.to(device=device),
            extrinsics=extrinsics.to(device=device),
            convention=convention,
            name=self._name,
            id=self._id,
            device=device,
        )

    def scale_intrinsics(
        self,
        resolution: Optional[Tuple[int, int]] = None,
        scale: Optional[
            Union[Union[int, float], Tuple[Union[int, float], Union[int, float]]]
        ] = None,
    ) -> "Camera":
        scaled_intrinsics = scale_intrinsics(
            intrinsics=self._intrinsics,
            resolution=resolution,
            scale=scale,
            inplace=False,
        )

        return Camera(
            intrinsics=scaled_intrinsics,
            extrinsics=self._extrinsics,
            convention=self._convention,
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
        def _validate_inputs() -> None:
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

        _validate_inputs()

        def _normalize_inputs(
            rotation: np.ndarray,
            translation: np.ndarray,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            rotation = torch.from_numpy(rotation).to(
                device=self._device,
                dtype=self._extrinsics.dtype,
            )
            translation = torch.from_numpy(translation).to(
                device=self._device,
                dtype=self._extrinsics.dtype,
            )
            assert isinstance(rotation, torch.Tensor), (
                "Expected normalized transform rotation to be a torch.Tensor. "
                f"{type(rotation)=}"
            )
            assert isinstance(translation, torch.Tensor), (
                "Expected normalized transform translation to be a torch.Tensor. "
                f"{type(translation)=}"
            )
            return rotation, translation

        rotation, translation = _normalize_inputs(
            rotation=rotation,
            translation=translation,
        )

        extrinsics = self._extrinsics
        R_c2w = extrinsics[:3, :3]
        t_c2w = extrinsics[:3, 3]
        R_c2w_new = rotation @ R_c2w
        t_c2w_new = scale * (rotation @ t_c2w) + translation

        extrinsics_new = torch.eye(
            4,
            dtype=extrinsics.dtype,
            device=extrinsics.device,
        )
        extrinsics_new[:3, :3] = R_c2w_new
        extrinsics_new[:3, 3] = t_c2w_new
        extrinsics_new[:3, :3] = _stabilize_rotation_matrix(extrinsics_new[:3, :3])

        return Camera(
            intrinsics=self._intrinsics,
            extrinsics=extrinsics_new,
            convention=self._convention,
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


def _stabilize_rotation_matrix(rotation: torch.Tensor) -> torch.Tensor:
    def _validate_inputs() -> None:
        assert isinstance(rotation, torch.Tensor), (
            "Expected rotation matrix to be a torch.Tensor. " f"{type(rotation)=}"
        )
        assert rotation.shape == (3, 3), (
            "Expected rotation matrix shape to be 3x3. " f"{rotation.shape=}"
        )
        assert rotation.dtype in (torch.float32, torch.float64), (
            "Expected rotation matrix dtype to be float32 or float64. "
            f"{rotation.dtype=}"
        )

    _validate_inputs()

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
