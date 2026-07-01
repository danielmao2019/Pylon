import math
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Optional, Tuple, Union

import torch

from data.structures.three_d.camera.intrinsics.scaling import (
    scale_camera_intrinsics_params,
)
from data.structures.three_d.camera.intrinsics.validation import (
    validate_camera_intrinsics_attributes,
)


class CameraIntrinsics(ABC):
    """Abstract base for a camera's intrinsics.

    Owns the named params plus device and the projection contract; each concrete
    subclass is exactly one camera model.
    """

    MODEL: ClassVar[str]

    def __init__(
        self,
        params: Dict[str, Union[int, float]],
        device: Union[str, torch.device] = torch.device("cuda"),
    ) -> None:
        """Construct a CameraIntrinsics from its model's named params and a device.

        Args:
            params: The model's named intrinsics parameters.
            device: Device the intrinsics live on, a string or torch.device.

        Returns:
            None.
        """
        validate_camera_intrinsics_attributes(
            model=type(self).MODEL,
            params=params,
            device=device,
        )
        self._params: Dict[str, Union[int, float]] = dict(params)
        self._device: torch.device = torch.device(device)

    @property
    def model(self) -> str:
        """The camera-model identifier.

        Args:
            None.

        Returns:
            The model identifier ``type(self).MODEL``.
        """
        return type(self).MODEL

    @property
    def params(self) -> Dict[str, Union[int, float]]:
        """The model's named intrinsics parameters.

        Args:
            None.

        Returns:
            The named intrinsics params.
        """
        return self._params

    @property
    def device(self) -> torch.device:
        """The device the intrinsics live on.

        Args:
            None.

        Returns:
            The device.
        """
        return self._device

    @property
    def cx(self) -> float:
        """The horizontal principal-point coordinate.

        Args:
            None.

        Returns:
            ``params["cx"]`` as a float.
        """
        return float(self._params["cx"])

    @property
    def cy(self) -> float:
        """The vertical principal-point coordinate.

        Args:
            None.

        Returns:
            ``params["cy"]`` as a float.
        """
        return float(self._params["cy"])

    @property
    @abstractmethod
    def fx(self) -> float:
        """The horizontal focal length / scale, whose params key differs per model.

        Args:
            None.

        Returns:
            The horizontal focal length / scale as a float.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def fy(self) -> float:
        """The vertical focal length / scale, whose params key differs per model.

        Args:
            None.

        Returns:
            The vertical focal length / scale as a float.
        """
        raise NotImplementedError

    @abstractmethod
    def project(
        self, points_camera: torch.Tensor, inplace: bool = False
    ) -> torch.Tensor:
        """Map camera-space 3D points to 2D image points under this model.

        Args:
            points_camera: Camera-space points, a ``[..., 3]`` torch.Tensor.
            inplace: If True, project in place — write the image points over the
                first two columns of ``points_camera`` and return a ``[..., 2]``
                view aliasing that input (its depth column is left intact). If
                False, return a freshly allocated ``[..., 2]`` and leave
                ``points_camera`` unchanged.

        Returns:
            The ``[..., 2]`` image points torch.Tensor (a view into
            ``points_camera`` when inplace, else a new tensor).
        """
        raise NotImplementedError

    def scale_intrinsics(
        self,
        resolution: Optional[Tuple[int, int]] = None,
        scale: Optional[
            Union[Union[int, float], Tuple[Union[int, float], Union[int, float]]]
        ] = None,
    ) -> "CameraIntrinsics":
        """Return this CameraIntrinsics scaled to a resolution or by a factor.

        Scales the focal params and principal-point params via
        ``scale_camera_intrinsics_params``. Exactly one of ``resolution`` or
        ``scale`` must be provided.

        Args:
            resolution: Optional target image resolution as ``(height, width)``.
            scale: Optional uniform scale, or a per-axis ``(sx, sy)`` tuple.

        Returns:
            A new CameraIntrinsics of the same model with scaled params.
        """
        scaled_params = scale_camera_intrinsics_params(
            params=self._params,
            resolution=resolution,
            scale=scale,
        )
        return type(self)(params=scaled_params, device=self._device)

    def to(
        self, device: Optional[Union[str, torch.device]] = None
    ) -> "CameraIntrinsics":
        """Return this CameraIntrinsics on a target device.

        Args:
            device: Target device; ``None`` keeps the current device.

        Returns:
            This CameraIntrinsics when the device is unchanged, else a new one.
        """
        assert device is None or isinstance(device, (str, torch.device)), (
            "Expected target device to be None, a string, or torch.device. "
            f"{device=}"
        )
        if device is None:
            return self
        target_device = torch.device(device)
        if target_device == self._device:
            return self
        return type(self)(params=self._params, device=target_device)


class CameraIntrinsicsSimplePinhole(CameraIntrinsics):
    """Simple-pinhole intrinsics: a single shared focal length f, perspective model."""

    MODEL: ClassVar[str] = "simple_pinhole"

    @property
    def fx(self) -> float:
        """The shared focal length.

        Args:
            None.

        Returns:
            ``params["f"]`` as a float.
        """
        return float(self._params["f"])

    @property
    def fy(self) -> float:
        """The shared focal length.

        Args:
            None.

        Returns:
            ``params["f"]`` as a float.
        """
        return float(self._params["f"])

    def project(
        self, points_camera: torch.Tensor, inplace: bool = False
    ) -> torch.Tensor:
        """Perspective projection with a single shared focal length.

        Args:
            points_camera: Camera-space points, a ``[..., 3]`` torch.Tensor.
            inplace: If True, project in place — write the image points over the
                first two columns of ``points_camera`` and return a ``[..., 2]``
                view aliasing that input (its depth column is left intact). If
                False, return a freshly allocated ``[..., 2]`` and leave
                ``points_camera`` unchanged.

        Returns:
            The ``[..., 2]`` image points torch.Tensor (a view into
            ``points_camera`` when inplace, else a new tensor).
        """
        assert isinstance(points_camera, torch.Tensor), (
            "Expected points_camera to be a torch.Tensor. " f"{type(points_camera)=}"
        )
        assert points_camera.shape[-1] == 3, (
            "Expected points_camera last dim to be 3. " f"{points_camera.shape=}"
        )
        assert isinstance(inplace, bool), (
            "Expected inplace to be a bool. " f"{type(inplace)=}"
        )
        out = points_camera[..., :2] if inplace else points_camera[..., :2].clone()
        z = points_camera[..., 2]
        f, cx, cy = float(self._params["f"]), self.cx, self.cy
        out[..., 0].div_(z).mul_(f).add_(cx)
        out[..., 1].div_(z).mul_(f).add_(cy)
        return out

    @property
    def fov(self) -> Tuple[float, float]:
        """The horizontal / vertical field of view in degrees.

        Args:
            None.

        Returns:
            The ``(horizontal, vertical)`` field of view in degrees.
        """
        focal = float(self._params["f"])
        horizontal_fov = 2.0 * math.atan(self.cx / focal) * 180.0 / math.pi
        vertical_fov = 2.0 * math.atan(self.cy / focal) * 180.0 / math.pi
        return (horizontal_fov, vertical_fov)


class CameraIntrinsicsPinhole(CameraIntrinsics):
    """Pinhole intrinsics: independent focal lengths fx / fy, perspective model."""

    MODEL: ClassVar[str] = "pinhole"

    @property
    def fx(self) -> float:
        """The horizontal focal length.

        Args:
            None.

        Returns:
            ``params["fx"]`` as a float.
        """
        return float(self._params["fx"])

    @property
    def fy(self) -> float:
        """The vertical focal length.

        Args:
            None.

        Returns:
            ``params["fy"]`` as a float.
        """
        return float(self._params["fy"])

    def project(
        self, points_camera: torch.Tensor, inplace: bool = False
    ) -> torch.Tensor:
        """Perspective projection with independent fx / fy.

        Args:
            points_camera: Camera-space points, a ``[..., 3]`` torch.Tensor.
            inplace: If True, project in place — write the image points over the
                first two columns of ``points_camera`` and return a ``[..., 2]``
                view aliasing that input (its depth column is left intact). If
                False, return a freshly allocated ``[..., 2]`` and leave
                ``points_camera`` unchanged.

        Returns:
            The ``[..., 2]`` image points torch.Tensor (a view into
            ``points_camera`` when inplace, else a new tensor).
        """
        assert isinstance(points_camera, torch.Tensor), (
            "Expected points_camera to be a torch.Tensor. " f"{type(points_camera)=}"
        )
        assert points_camera.shape[-1] == 3, (
            "Expected points_camera last dim to be 3. " f"{points_camera.shape=}"
        )
        assert isinstance(inplace, bool), (
            "Expected inplace to be a bool. " f"{type(inplace)=}"
        )
        out = points_camera[..., :2] if inplace else points_camera[..., :2].clone()
        z = points_camera[..., 2]
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        out[..., 0].div_(z).mul_(fx).add_(cx)
        out[..., 1].div_(z).mul_(fy).add_(cy)
        return out

    @property
    def fov(self) -> Tuple[float, float]:
        """The horizontal / vertical field of view in degrees.

        Args:
            None.

        Returns:
            The ``(horizontal, vertical)`` field of view in degrees.
        """
        horizontal_fov = 2.0 * math.atan(self.cx / self.fx) * 180.0 / math.pi
        vertical_fov = 2.0 * math.atan(self.cy / self.fy) * 180.0 / math.pi
        return (horizontal_fov, vertical_fov)


class CameraIntrinsicsOrtho(CameraIntrinsics):
    """Ortho (weak-perspective) intrinsics: focal scales fx / fy, no perspective divide."""

    MODEL: ClassVar[str] = "ortho"

    @property
    def fx(self) -> float:
        """The horizontal focal scale.

        Args:
            None.

        Returns:
            ``params["fx"]`` as a float.
        """
        return float(self._params["fx"])

    @property
    def fy(self) -> float:
        """The vertical focal scale.

        Args:
            None.

        Returns:
            ``params["fy"]`` as a float.
        """
        return float(self._params["fy"])

    def project(
        self, points_camera: torch.Tensor, inplace: bool = False
    ) -> torch.Tensor:
        """Orthographic projection: scale and offset without the perspective divide.

        Args:
            points_camera: Camera-space points, a ``[..., 3]`` torch.Tensor.
            inplace: If True, project in place — write the image points over the
                first two columns of ``points_camera`` and return a ``[..., 2]``
                view aliasing that input (its depth column is left intact). If
                False, return a freshly allocated ``[..., 2]`` and leave
                ``points_camera`` unchanged.

        Returns:
            The ``[..., 2]`` image points torch.Tensor (a view into
            ``points_camera`` when inplace, else a new tensor).
        """
        assert isinstance(points_camera, torch.Tensor), (
            "Expected points_camera to be a torch.Tensor. " f"{type(points_camera)=}"
        )
        assert points_camera.shape[-1] == 3, (
            "Expected points_camera last dim to be 3. " f"{points_camera.shape=}"
        )
        assert isinstance(inplace, bool), (
            "Expected inplace to be a bool. " f"{type(inplace)=}"
        )
        out = points_camera[..., :2] if inplace else points_camera[..., :2].clone()
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        out[..., 0].mul_(fx).add_(cx)
        out[..., 1].mul_(fy).add_(cy)
        return out


def build_camera_intrinsics(
    model: str,
    params: Dict[str, Union[int, float]],
    device: Union[str, torch.device] = torch.device("cuda"),
) -> CameraIntrinsics:
    """Build the CameraIntrinsics subclass for a camera-model string.

    The serialization-boundary factory; dispatches on the model identifier.

    Args:
        model: Camera-model identifier string.
        params: The model's named intrinsics parameters.
        device: Device the intrinsics live on, a string or torch.device.

    Returns:
        The CameraIntrinsics subclass instance for the model.
    """
    if model == "simple_pinhole":
        return CameraIntrinsicsSimplePinhole(params=params, device=device)
    if model == "pinhole":
        return CameraIntrinsicsPinhole(params=params, device=device)
    if model == "ortho":
        return CameraIntrinsicsOrtho(params=params, device=device)
    assert 0, "Should not reach here. " f"{model=}"
