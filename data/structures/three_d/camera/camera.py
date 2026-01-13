import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from data.structures.three_d.camera.conventions import _transform_convention


class Camera:

    def __init__(
        self,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        convention: str,
        name: str | None = None,
        id: int | None = None,
        device: str | torch.device = torch.device('cuda'),
    ) -> None:
        # Input validations
        Camera._validate_camera_intrinsics(intrinsics)
        Camera._validate_camera_extrinsics(extrinsics)
        Camera._validate_convention(convention)
        assert name is None or isinstance(name, str), f"{type(name)=}"
        assert id is None or isinstance(id, int), f"{type(id)=}"

        target_device = torch.device(device)
        intrinsics = intrinsics.to(device=target_device)
        extrinsics = extrinsics.to(device=target_device)

        self._intrinsics = intrinsics
        self._extrinsics = extrinsics
        self._convention = convention
        self._name = name
        self._id = id
        self._device = target_device

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
    def name(self) -> str | None:
        return self._name

    @property
    def id(self) -> int | None:
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

    def to(
        self, device: str | torch.device | None = None, convention: str | None = None
    ) -> "Camera":
        target_device = torch.device(device) if device is not None else self._device
        target_convention = convention if convention is not None else self._convention
        Camera._validate_convention(target_convention)

        extrinsics = (
            _transform_convention(
                camera=self,
                target_convention=target_convention,
            )
            if target_convention != self._convention
            else self._extrinsics
        )

        return Camera(
            intrinsics=self._intrinsics.to(device=target_device),
            extrinsics=extrinsics.to(device=target_device),
            convention=target_convention,
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
        from data.structures.three_d.camera.scaling import scale_intrinsics

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

    @classmethod
    def from_serialized(
        cls, payload: Dict[str, Any], device: str | torch.device | None = None
    ) -> "Camera":
        # Input validations
        assert isinstance(payload, dict), f"{type(payload)=}"
        assert "intrinsics" in payload, f"{payload.keys()=}"
        assert "extrinsics" in payload, f"{payload.keys()=}"
        assert "convention" in payload, f"{payload.keys()=}"
        assert "name" in payload, f"{payload.keys()=}"
        assert "id" in payload, f"{payload.keys()=}"
        assert device is None or isinstance(device, (str, torch.device)), f"{device=}"

        target_device = torch.device(device) if device is not None else None
        intrinsics = torch.tensor(
            payload["intrinsics"],
            dtype=torch.float32,
            device=target_device,
        )
        extrinsics = torch.tensor(
            payload["extrinsics"],
            dtype=torch.float32,
            device=target_device,
        )
        convention = payload["convention"]
        name = payload["name"]
        camera_id = payload["id"]

        return cls(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            convention=convention,
            name=name,
            id=camera_id,
            device=target_device if target_device is not None else intrinsics.device,
        )

    def to_serialized(self) -> Dict[str, Any]:
        # Input validations
        assert isinstance(self, Camera), f"{type(self)=}"

        return {
            "intrinsics": self._intrinsics.detach().cpu().tolist(),
            "extrinsics": self._extrinsics.detach().cpu().tolist(),
            "convention": self._convention,
            "name": self._name,
            "id": self._id,
        }

    @staticmethod
    def _check_camera_intrinsics_numpy(obj: Any) -> None:
        assert isinstance(obj, np.ndarray), "Camera intrinsics must be a numpy array."
        assert obj.ndim == 2, "Camera intrinsics must be a 2D array."
        assert obj.shape == (3, 3), "Camera intrinsics must be of shape (3, 3)."
        assert obj.dtype == np.float32, "Camera intrinsics must be of type float32."
        fx, fy = obj[0, 0], obj[1, 1]
        assert (
            fx > 0 and fy > 0
        ), "Focal lengths (elements [0, 0] and [1, 1]) must be positive."
        cx, cy = obj[0, 2], obj[1, 2]
        assert (
            cx >= 0 and cy >= 0
        ), "Principal point coordinates (elements [0, 2] and [1, 2]) must be non-negative."
        assert (
            obj[0, 1] == 0
        ), "Camera intrinsics must have zero skew (element [0, 1] must be 0)."
        assert (
            obj[1, 0] == 0
        ), "Camera intrinsics must have zero skew (element [1, 0] must be 0)."
        assert np.array_equal(
            obj[2, :], np.array([0, 0, 1])
        ), "Camera intrinsics must have [0, 0, 1] in the last row."

    @staticmethod
    def _check_camera_intrinsics_torch(obj: Any) -> None:
        assert isinstance(
            obj, torch.Tensor
        ), "Camera intrinsics must be a torch tensor."
        assert obj.ndim == 2, "Camera intrinsics must be a 2D tensor."
        assert obj.shape == (3, 3), "Camera intrinsics must be of shape (3, 3)."
        assert obj.dtype == torch.float32, "Camera intrinsics must be of type float32."
        fx, fy = obj[0, 0], obj[1, 1]
        assert (
            fx > 0 and fy > 0
        ), "Focal lengths (elements [0, 0] and [1, 1]) must be positive."
        cx, cy = obj[0, 2], obj[1, 2]
        assert (
            cx >= 0 and cy >= 0
        ), "Principal point coordinates (elements [0, 2] and [1, 2]) must be non-negative."
        assert (
            obj[0, 1] == 0
        ), "Camera intrinsics must have zero skew (element [0, 1] must be 0)."
        assert (
            obj[1, 0] == 0
        ), "Camera intrinsics must have zero skew (element [1, 0] must be 0)."
        assert torch.equal(
            obj[2, :], torch.tensor([0, 0, 1], dtype=obj.dtype, device=obj.device)
        ), "Camera intrinsics must have [0, 0, 1] in the last row."

    @staticmethod
    def _validate_camera_intrinsics(obj: Any) -> None:
        if isinstance(obj, np.ndarray):
            Camera._check_camera_intrinsics_numpy(obj)
        elif isinstance(obj, torch.Tensor):
            Camera._check_camera_intrinsics_torch(obj)
        else:
            raise TypeError(
                "Camera intrinsics must be a numpy array or a torch tensor."
            )

    @staticmethod
    def _check_rotation_matrix_numpy(obj: Any) -> None:
        assert isinstance(obj, np.ndarray), "Rotation matrix must be a numpy array."
        assert obj.ndim == 2, "Rotation matrix must be a 2D array."
        assert obj.shape == (3, 3), "Rotation matrix must be of shape (3, 3)."
        assert obj.dtype == np.float32, "Rotation matrix must be of type float32."

        should_be_identity = obj @ obj.T
        max_diff = np.max(np.abs(should_be_identity - np.eye(3)))
        assert np.allclose(
            should_be_identity,
            np.eye(3),
            atol=1.0e-06,
            rtol=0.0,
        ), "Rotation matrix must be orthogonal. Max diff between RR^T and I: {:.6g}".format(
            max_diff
        )

        det = np.linalg.det(obj)
        assert np.isclose(
            det,
            1.0,
            atol=1.0e-06,
            rtol=0.0,
        ), f"Rotation matrix must have determinant +1. det(R) = {det:.6g}."

    @staticmethod
    def _check_camera_extrinsics_numpy(obj: Any) -> None:
        assert isinstance(obj, np.ndarray), "Camera extrinsics must be a numpy array."
        assert obj.ndim == 2, "Camera extrinsics must be a 2D array."
        assert obj.shape == (4, 4), "Camera extrinsics must be of shape (4, 4)."
        assert obj.dtype == np.float32, "Camera extrinsics must be of type float32."
        assert np.allclose(
            obj[3, :], np.array([0, 0, 0, 1])
        ), "Camera extrinsics must have [0, 0, 0, 1] in the last row."
        rotation = obj[:3, :3]
        Camera._check_rotation_matrix_numpy(rotation)

    @staticmethod
    def _check_rotation_matrix_torch(obj: Any) -> None:
        assert isinstance(obj, torch.Tensor), "Rotation matrix must be a torch tensor."
        assert obj.ndim == 2, "Rotation matrix must be a 2D tensor."
        assert obj.shape == (3, 3), "Rotation matrix must be of shape (3, 3)."
        assert obj.dtype == torch.float32, "Rotation matrix must be of type float32."

        identity = torch.eye(3, dtype=obj.dtype, device=obj.device)
        should_be_identity = obj @ obj.T
        max_diff = torch.max(torch.abs(should_be_identity - identity))
        assert torch.allclose(
            should_be_identity,
            identity,
            atol=1.0e-06,
            rtol=0.0,
        ), (
            "Rotation matrix must be orthogonal. Max diff between RR^T and I: "
            f"{float(max_diff)}"
        )

        det = torch.det(obj)
        assert torch.isclose(
            det,
            torch.tensor(1.0, dtype=obj.dtype, device=obj.device),
            atol=1.0e-06,
            rtol=0.0,
        ), f"Rotation matrix must have determinant +1. det(R) = {float(det)}."

    @staticmethod
    def _check_camera_extrinsics_torch(obj: Any) -> None:
        assert isinstance(
            obj, torch.Tensor
        ), "Camera extrinsics must be a torch tensor."
        assert obj.ndim == 2, "Camera extrinsics must be a 2D tensor."
        assert obj.shape == (4, 4), "Camera extrinsics must be of shape (4, 4)."
        assert obj.dtype == torch.float32, "Camera extrinsics must be of type float32."
        assert torch.equal(
            obj[3, :], torch.tensor([0, 0, 0, 1], dtype=obj.dtype, device=obj.device)
        ), "Camera extrinsics must have [0, 0, 0, 1] in the last row."
        rotation = obj[:3, :3]
        Camera._check_rotation_matrix_torch(rotation)

    @staticmethod
    def _validate_camera_extrinsics(obj: Any) -> None:
        if isinstance(obj, np.ndarray):
            Camera._check_camera_extrinsics_numpy(obj)
        elif isinstance(obj, torch.Tensor):
            Camera._check_camera_extrinsics_torch(obj)
        else:
            raise TypeError(
                "Camera extrinsics must be a numpy array or a torch tensor."
            )

    @staticmethod
    def _validate_rotation_matrix(obj: Any) -> None:
        if isinstance(obj, np.ndarray):
            Camera._check_rotation_matrix_numpy(obj)
        elif isinstance(obj, torch.Tensor):
            Camera._check_rotation_matrix_torch(obj)
        else:
            raise TypeError("Rotation matrix must be a numpy array or a torch tensor.")

    @staticmethod
    def _validate_convention(convention: str) -> None:
        assert isinstance(convention, str), f"{type(convention)=}"
        assert convention in [
            "opengl",
            "standard",
            "opencv",
            "pytorch3d",
        ], f"Unsupported convention: {convention}"
