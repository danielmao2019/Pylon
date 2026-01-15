import math
from typing import Any, Dict, Optional, Tuple, Union

import torch

from data.structures.three_d.camera.conventions import _transform_convention
from data.structures.three_d.camera.validation import (
    validate_camera_convention,
    validate_camera_extrinsics,
    validate_camera_intrinsics,
)


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
        assert intrinsics is None or isinstance(intrinsics, torch.Tensor), f"{type(intrinsics)=}"
        validate_camera_extrinsics(extrinsics)
        validate_camera_convention(convention)
        assert name is None or isinstance(name, str), f"{type(name)=}"
        assert id is None or isinstance(id, int), f"{type(id)=}"

        # Input normalization
        if intrinsics is None:
            intrinsics = torch.eye(3, dtype=torch.float32, device=device)
        else:
            intrinsics = intrinsics.to(device=device)
        validate_camera_intrinsics(intrinsics)
        extrinsics = extrinsics.to(device=device)
        device = torch.device(device)

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

    @property
    def right(self) -> torch.Tensor:
        if self._convention == "standard":
            vec = self._extrinsics[:3, 0]
        elif self._convention == "opengl":
            vec = self._extrinsics[:3, 0]
        elif self._convention == "opencv":
            vec = self._extrinsics[:3, 0]
        elif self._convention == "pytorch3d":
            vec = self._extrinsics[:3, 0]
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
            vec = self._extrinsics[:3, 1]
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
            vec = self._extrinsics[:3, 2]
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
        self, device: str | torch.device | None = None, convention: str | None = None
    ) -> "Camera":
        target_device = torch.device(device) if device is not None else self._device
        target_convention = convention if convention is not None else self._convention
        validate_camera_convention(target_convention)

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
