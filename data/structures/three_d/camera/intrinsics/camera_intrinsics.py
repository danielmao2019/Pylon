import math
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Optional, Tuple, Union

import torch

from data.structures.three_d.camera.intrinsics.conventions import (
    transform_intr_convention,
)
from data.structures.three_d.camera.intrinsics.scaling import resolve_target_resolution
from data.structures.three_d.camera.intrinsics.validation import (
    validate_camera_intrinsics_attributes,
    validate_intr_convention,
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
        intr_convention: str,
        device: Union[str, torch.device] = torch.device("cuda"),
    ) -> None:
        """Construct a CameraIntrinsics from its model's named params, the image-plane frame they are stated in, and a device.

        Args:
            params: The model's named intrinsics parameters; carries the resolution keys ``h`` / ``w`` alongside the projection keys.
            intr_convention: Image-plane frame the params are stated in, one of ``standard`` / ``opengl`` / ``pytorch3d`` / ``vulkan``.
            device: Device the intrinsics live on, a string or torch.device.

        Returns:
            None.
        """
        validate_camera_intrinsics_attributes(
            model=type(self).MODEL,
            intr_convention=intr_convention,
            params=params,
            device=device,
        )
        self._params: Dict[str, Union[int, float]] = dict(params)
        self._intr_convention: str = intr_convention
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
    def intr_convention(self) -> str:
        """The image-plane frame these params are stated in.

        Args:
            None.

        Returns:
            The image-plane convention string (standard / opengl / pytorch3d /
            vulkan), without which a principal point names no location.
        """
        return self._intr_convention

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
    def resolution(self) -> Tuple[int, int]:
        """The resolution these params are stated against.

        Args:
            None.

        Returns:
            The ``(height, width)`` pair read off the two params that carry it,
            since a principal point in pixels names a location only against them.
        """
        return self._params["h"], self._params["w"]

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

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        intr_convention: Optional[str] = None,
    ) -> "CameraIntrinsics":
        """Return this CameraIntrinsics on a target device / image-plane frame.

        The intrinsics half of the frame change its extrinsics counterpart
        performs on the pose.

        Args:
            device: Target device; ``None`` keeps the current device.
            intr_convention: Target image-plane frame; ``None`` keeps the current one.

        Returns:
            A fresh CameraIntrinsics of the same model on the target device and
            image-plane frame.
        """

        def _validate_inputs() -> None:
            assert device is None or isinstance(device, (str, torch.device)), (
                "Expected target device to be None, a string, or torch.device. "
                f"{device=}"
            )
            if intr_convention is not None:
                validate_intr_convention(intr_convention=intr_convention)

        _validate_inputs()

        params = self._params
        if intr_convention is not None and intr_convention != self._intr_convention:
            params = transform_intr_convention(
                params=params,
                model=type(self).MODEL,
                source_intr_convention=self._intr_convention,
                target_intr_convention=intr_convention,
            )
        target_device = torch.device(device) if device is not None else self._device
        return type(self)(
            params=params,
            intr_convention=intr_convention or self._intr_convention,
            device=target_device,
        )

    def transform_intrinsics(
        self,
        transform: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> "CameraIntrinsics":
        """Return this CameraIntrinsics restated onto another image by a pixel-frame affine.

        The raster that image is named alongside it, because a 3x3 carries no size
        of its own.

        Args:
            transform: Pixel-frame affine as a ``(3, 3)`` float32 torch.Tensor whose last row is ``[0, 0, 1]``.
            resolution: The target image's own resolution as ``(height, width)``.

        Returns:
            A new CameraIntrinsics of the same model, on this intrinsics' own
            image-plane frame, stated against ``resolution``.
        """

        def _validate_inputs() -> None:
            assert isinstance(transform, torch.Tensor), (
                "Expected the intrinsics transform to be a torch.Tensor. "
                f"{type(transform)=}"
            )
            assert transform.shape == (3, 3), (
                "Expected the intrinsics transform to be (3, 3). " f"{transform.shape=}"
            )
            assert transform.dtype == torch.float32, (
                "Expected the intrinsics transform dtype to be float32. "
                f"{transform.dtype=}"
            )
            assert torch.equal(
                transform[2].detach().cpu(),
                torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
            ), (
                "Expected the intrinsics transform last row to be [0, 0, 1]. "
                f"{transform[2]=}"
            )
            assert isinstance(resolution, tuple) and len(resolution) == 2, (
                "Expected resolution to be a (height, width) tuple of length 2. "
                f"{resolution=}"
            )
            assert isinstance(resolution[0], int) and isinstance(resolution[1], int), (
                "Expected resolution values to be integers (height, width). "
                f"{resolution=}"
            )
            assert resolution[0] > 0 and resolution[1] > 0, (
                "Expected resolution values to be positive integers. " f"{resolution=}"
            )

        _validate_inputs()

        params = transform_intr_convention(
            params=self._params,
            model=type(self).MODEL,
            source_intr_convention=self._intr_convention,
            target_intr_convention="standard",
        )
        standard = type(self)(
            params=params,
            intr_convention="standard",
            device=self._device,
        )
        source_matrix = torch.zeros(
            (3, 3),
            dtype=torch.float32,
            device=transform.device,
        )
        source_matrix[0, 0] = standard.fx
        source_matrix[1, 1] = standard.fy
        source_matrix[0, 2] = standard.cx
        source_matrix[1, 2] = standard.cy
        source_matrix[2, 2] = 1.0
        target_matrix = transform @ source_matrix
        if type(self).MODEL == "simple_pinhole":
            assert torch.equal(target_matrix[0, 0], target_matrix[1, 1]), (
                "Expected the affine to scale both image axes alike for "
                "simple_pinhole, whose one shared f holds one ratio. "
                f"{target_matrix[0, 0]=} {target_matrix[1, 1]=}"
            )
        if type(self).MODEL == "simple_pinhole":
            params = {"f": float(target_matrix[0, 0])}
        else:
            params = {
                "fx": float(target_matrix[0, 0]),
                "fy": float(target_matrix[1, 1]),
            }
        params["cx"] = float(target_matrix[0, 2])
        params["cy"] = float(target_matrix[1, 2])
        params["h"] = resolution[0]
        params["w"] = resolution[1]
        params = transform_intr_convention(
            params=params,
            model=type(self).MODEL,
            source_intr_convention="standard",
            target_intr_convention=self._intr_convention,
        )
        return type(self)(
            params=params,
            intr_convention=self._intr_convention,
            device=self._device,
        )

    def scale_intrinsics(
        self,
        resolution: Optional[Tuple[int, int]] = None,
        scale: Optional[
            Union[Union[int, float], Tuple[Union[int, float], Union[int, float]]]
        ] = None,
    ) -> "CameraIntrinsics":
        """Return this CameraIntrinsics restated against a different resolution.

        The diagonal case of an intrinsics transform, so this builds that transform
        and the one owner applies it. Exactly one of ``resolution`` or ``scale``
        must be provided.

        Args:
            resolution: Optional target image resolution as ``(height, width)``.
            scale: Optional uniform scale, or a per-axis ``(sx, sy)`` tuple.

        Returns:
            A new CameraIntrinsics of the same model stated against the target
            resolution.
        """

        def _validate_inputs() -> None:
            assert (resolution is None) ^ (scale is None), (
                "Expected exactly one of resolution or scale to be provided. "
                f"{resolution=} {scale=}"
            )

        _validate_inputs()

        def _normalize_inputs() -> Tuple[int, int]:
            return resolve_target_resolution(
                params=self._params,
                resolution=resolution,
                scale=scale,
            )

        resolution = _normalize_inputs()

        scale_x = float(resolution[1]) / float(self._params["w"])
        scale_y = float(resolution[0]) / float(self._params["h"])
        transform = torch.tensor(
            [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self._device,
        )
        return self.transform_intrinsics(transform=transform, resolution=resolution)


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
    intr_convention: str,
    device: Union[str, torch.device] = torch.device("cuda"),
) -> CameraIntrinsics:
    """Build the CameraIntrinsics subclass for a camera-model string.

    The serialization-boundary factory; dispatches on the model identifier.

    Args:
        model: Camera-model identifier string.
        params: The model's named intrinsics parameters; carries the resolution keys ``h`` / ``w`` alongside the projection keys.
        intr_convention: Image-plane frame the params are stated in, one of ``standard`` / ``opengl`` / ``pytorch3d`` / ``vulkan``.
        device: Device the intrinsics live on, a string or torch.device.

    Returns:
        The CameraIntrinsics subclass instance for the model.
    """
    if model == "simple_pinhole":
        return CameraIntrinsicsSimplePinhole(
            params=params,
            intr_convention=intr_convention,
            device=device,
        )
    if model == "pinhole":
        return CameraIntrinsicsPinhole(
            params=params,
            intr_convention=intr_convention,
            device=device,
        )
    if model == "ortho":
        return CameraIntrinsicsOrtho(
            params=params,
            intr_convention=intr_convention,
            device=device,
        )
    assert 0, "Should not reach here. " f"{model=}"
