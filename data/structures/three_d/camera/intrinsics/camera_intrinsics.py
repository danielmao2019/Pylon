import math
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
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
        params: Dict[str, Union[int, float, np.ndarray, torch.Tensor]],
        intr_convention: str,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Construct a CameraIntrinsics from tensor-compatible named scalar params.

        Args:
            params: The model's named scalar intrinsics parameters; carries the resolution keys ``h`` / ``w`` alongside the projection keys.
            intr_convention: Image-plane frame the params are stated in, one of ``standard`` / ``opengl`` / ``pytorch3d`` / ``vulkan``.
            device: Optional target device for the scalar tensor params.
            dtype: Optional target floating dtype for the scalar tensor params.

        Returns:
            None.
        """

        def _validate_inputs() -> None:
            assert isinstance(params, dict), (
                "Expected intrinsics params to be a dict. " f"{type(params)=}"
            )
            for key, value in params.items():
                assert isinstance(key, str), (
                    "Expected every intrinsics params key to be a string. "
                    f"{key=} {type(key)=}"
                )
                assert isinstance(value, (int, float, np.ndarray, torch.Tensor)), (
                    "Expected every intrinsics params value to be scalar-compatible. "
                    f"{key=} {type(value)=}"
                )
                if isinstance(value, np.ndarray):
                    assert value.size == 1, (
                        "Expected every numpy intrinsics param to contain one value. "
                        f"{key=} {value.shape=}"
                    )
                    assert np.issubdtype(value.dtype, np.number), (
                        "Expected every numpy intrinsics param to be numeric. "
                        f"{key=} {value.dtype=}"
                    )
                if isinstance(value, torch.Tensor):
                    assert value.numel() == 1, (
                        "Expected every tensor intrinsics param to contain one value. "
                        f"{key=} {value.shape=}"
                    )
                    assert not value.is_complex(), (
                        "Expected every tensor intrinsics param to be real-valued. "
                        f"{key=} {value.dtype=}"
                    )
            validate_intr_convention(intr_convention=intr_convention)
            assert device is None or isinstance(device, (str, torch.device)), (
                "Expected CameraIntrinsics device to be None, a string, or torch.device. "
                f"{type(device)=}"
            )
            assert dtype is None or isinstance(dtype, torch.dtype), (
                "Expected CameraIntrinsics dtype to be None or a torch dtype. "
                f"{type(dtype)=}"
            )
            if dtype is not None:
                assert torch.empty((), dtype=dtype).is_floating_point(), (
                    "Expected CameraIntrinsics dtype to be floating. " f"{dtype=}"
                )

        _validate_inputs()

        def _normalize_inputs(
            params: Dict[str, Union[int, float, np.ndarray, torch.Tensor]],
        ) -> Dict[str, torch.Tensor]:
            target_device = torch.device("cpu")
            for value in params.values():
                if isinstance(value, torch.Tensor):
                    target_device = value.device
                    break

            target_dtype = torch.get_default_dtype()
            for value in params.values():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    target_dtype = value.dtype
                    break
                if isinstance(value, np.ndarray) and np.issubdtype(
                    value.dtype, np.floating
                ):
                    target_dtype = torch.as_tensor(value.reshape(-1)[0]).dtype
                    break

            params = {
                key: torch.as_tensor(value)
                .reshape(())
                .to(device=target_device, dtype=target_dtype)
                for key, value in params.items()
            }
            for key, value in params.items():
                assert value.shape == (), (
                    "Expected every normalized intrinsics param to be scalar. "
                    f"{key=} {value.shape=}"
                )
                assert value.device == target_device, (
                    "Expected every normalized intrinsics param to share device. "
                    f"{key=} {value.device=} {target_device=}"
                )
                assert value.dtype == target_dtype, (
                    "Expected every normalized intrinsics param to share dtype. "
                    f"{key=} {value.dtype=} {target_dtype=}"
                )
            return params

        params = _normalize_inputs(params=params)
        validate_camera_intrinsics_attributes(
            model=type(self).MODEL,
            intr_convention=intr_convention,
            params=params,
            device=device,
            dtype=dtype,
        )

        self._params: Dict[str, torch.Tensor] = params
        self._intr_convention: str = intr_convention
        self._device: torch.device = next(iter(params.values())).device
        self._dtype: torch.dtype = next(iter(params.values())).dtype
        if device is not None or dtype is not None:
            intrinsics = self.to(device=device, dtype=dtype)
            self._params = intrinsics.params
            self._intr_convention = intrinsics.intr_convention
            self._device = intrinsics.device
            self._dtype = intrinsics.dtype

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
    def params(self) -> Dict[str, torch.Tensor]:
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
    def dtype(self) -> torch.dtype:
        """The dtype shared by the intrinsics params.

        Args:
            None.

        Returns:
            The torch dtype shared by every scalar param tensor.
        """
        return self._dtype

    @property
    def cx(self) -> torch.Tensor:
        """The horizontal principal-point coordinate.

        Args:
            None.

        Returns:
            The scalar tensor ``params["cx"]``.
        """
        return self._params["cx"]

    @property
    def cy(self) -> torch.Tensor:
        """The vertical principal-point coordinate.

        Args:
            None.

        Returns:
            The scalar tensor ``params["cy"]``.
        """
        return self._params["cy"]

    @property
    def resolution(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The resolution these params are stated against.

        Args:
            None.

        Returns:
            The ``(height, width)`` scalar tensor pair read off the two params that
            carry it, since a principal point in pixels names a location only
            against them.
        """
        return self._params["h"], self._params["w"]

    @property
    @abstractmethod
    def fx(self) -> torch.Tensor:
        """The horizontal focal length / scale, whose params key differs per model.

        Args:
            None.

        Returns:
            The horizontal focal length / scale as a scalar tensor.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def fy(self) -> torch.Tensor:
        """The vertical focal length / scale, whose params key differs per model.

        Args:
            None.

        Returns:
            The vertical focal length / scale as a scalar tensor.
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
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
        intr_convention: Optional[str] = None,
    ) -> "CameraIntrinsics":
        """Return this CameraIntrinsics with tensor placement and image-plane frame changes.

        The intrinsics half of the frame change its extrinsics counterpart performs
        on the pose.

        Args:
            device: Target device; ``None`` keeps the current device.
            dtype: Target floating dtype; ``None`` keeps the current dtype.
            non_blocking: Whether tensor moves may be asynchronous.
            copy: Whether tensor moves must allocate new storage even when unchanged.
            intr_convention: Target image-plane frame; ``None`` keeps the current one.

        Returns:
            This CameraIntrinsics when unchanged, else a CameraIntrinsics of the same
            model on the target device, dtype, and image-plane frame.
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
            if intr_convention is not None:
                validate_intr_convention(intr_convention=intr_convention)

        _validate_inputs()

        target_device = torch.device(device) if device is not None else self._device
        target_dtype = dtype if dtype is not None else self._dtype
        target_intr_convention = intr_convention or self._intr_convention
        if (
            target_device == self._device
            and target_dtype == self._dtype
            and target_intr_convention == self._intr_convention
            and copy is False
        ):
            return self

        params = self._params
        if intr_convention is not None and intr_convention != self._intr_convention:
            params = transform_intr_convention(
                params=params,
                model=type(self).MODEL,
                source_intr_convention=self._intr_convention,
                target_intr_convention=intr_convention,
            )
        params = {
            key: value.to(
                device=target_device,
                dtype=target_dtype,
                non_blocking=non_blocking,
                copy=copy,
            )
            for key, value in params.items()
        }
        return type(self)(
            params=params,
            intr_convention=target_intr_convention,
        )

    def transform_intrinsics(
        self,
        transform: torch.Tensor,
        resolution: Tuple[Union[int, torch.Tensor], Union[int, torch.Tensor]],
    ) -> "CameraIntrinsics":
        """Return this CameraIntrinsics restated onto another image by a pixel-frame affine.

        The raster that image is named alongside it, because a 3x3 carries no size
        of its own.

        Args:
            transform: Pixel-frame affine as a ``(3, 3)`` float32 torch.Tensor whose last row is ``[0, 0, 1]``.
            resolution: The target image's own resolution as ``(height, width)`` integer values or scalar integer-valued tensors.

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
            assert transform.is_floating_point(), (
                "Expected the intrinsics transform dtype to be floating. "
                f"{transform.dtype=}"
            )
            assert torch.equal(
                transform[2].detach().cpu(),
                torch.tensor([0.0, 0.0, 1.0], dtype=transform.dtype),
            ), (
                "Expected the intrinsics transform last row to be [0, 0, 1]. "
                f"{transform[2]=}"
            )
            assert isinstance(resolution, tuple) and len(resolution) == 2, (
                "Expected resolution to be a (height, width) tuple of length 2. "
                f"{resolution=}"
            )
            for value in resolution:
                assert isinstance(value, (int, torch.Tensor)), (
                    "Expected resolution values to be integers or scalar tensors. "
                    f"{type(value)=} {resolution=}"
                )
                if isinstance(value, torch.Tensor):
                    assert value.numel() == 1, (
                        "Expected tensor resolution values to contain one value. "
                        f"{value.shape=}"
                    )
                    assert torch.equal(value, torch.round(value)), (
                        "Expected tensor resolution values to be integer-valued. "
                        f"{value=}"
                    )
                    assert value > 0, (
                        "Expected tensor resolution values to be positive. " f"{value=}"
                    )
                else:
                    assert value > 0, (
                        "Expected resolution values to be positive integers. "
                        f"{resolution=}"
                    )

        _validate_inputs()

        def _normalize_inputs(
            transform: torch.Tensor,
            resolution: Tuple[Union[int, torch.Tensor], Union[int, torch.Tensor]],
        ) -> Tuple[torch.Tensor, Tuple[int, int]]:
            transform = transform.to(device=self._device, dtype=self._dtype)
            resolution = tuple(
                (
                    int(value.detach().cpu().item())
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for value in resolution
            )
            return transform, resolution

        transform, resolution = _normalize_inputs(
            transform=transform,
            resolution=resolution,
        )

        params = transform_intr_convention(
            params=self._params,
            model=type(self).MODEL,
            source_intr_convention=self._intr_convention,
            target_intr_convention="standard",
        )
        standard = type(self)(
            params=params,
            intr_convention="standard",
        )
        source_matrix = torch.zeros(
            (3, 3),
            dtype=self._dtype,
            device=self._device,
        )
        source_matrix[0, 0] = standard.fx
        source_matrix[1, 1] = standard.fy
        source_matrix[0, 2] = standard.cx
        source_matrix[1, 2] = standard.cy
        source_matrix[2, 2] = 1.0
        target_matrix = transform @ source_matrix
        if type(self).MODEL == "simple_pinhole":
            assert torch.isclose(target_matrix[0, 0], target_matrix[1, 1]), (
                "Expected the affine to scale both image axes alike for "
                "simple_pinhole, whose one shared f holds one ratio. "
                f"{target_matrix[0, 0]=} {target_matrix[1, 1]=}"
            )
        if type(self).MODEL == "simple_pinhole":
            params = {"f": target_matrix[0, 0]}
        else:
            params = {
                "fx": target_matrix[0, 0],
                "fy": target_matrix[1, 1],
            }
        params["cx"] = target_matrix[0, 2]
        params["cy"] = target_matrix[1, 2]
        params["h"] = torch.as_tensor(
            resolution[0],
            dtype=self._dtype,
            device=self._device,
        )
        params["w"] = torch.as_tensor(
            resolution[1],
            dtype=self._dtype,
            device=self._device,
        )
        params = transform_intr_convention(
            params=params,
            model=type(self).MODEL,
            source_intr_convention="standard",
            target_intr_convention=self._intr_convention,
        )
        return type(self)(
            params=params,
            intr_convention=self._intr_convention,
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
    ) -> "CameraIntrinsics":
        """Return this CameraIntrinsics restated against a different resolution.

        The diagonal case of an intrinsics transform, so this builds that transform
        and the one owner applies it. Exactly one of ``resolution`` or ``scale``
        must be provided.

        Args:
            resolution: Optional target image resolution as one integer side or ``(height, width)``.
            scale: Optional uniform scale, or a per-axis ``(sx, sy)`` pair.

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

        def _normalize_inputs(
            resolution: Optional[
                Union[int, Tuple[int, int], List[int], np.ndarray, torch.Tensor]
            ],
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
            ],
        ) -> Tuple[
            Tuple[int, int],
            Optional[Tuple[torch.Tensor, torch.Tensor]],
        ]:
            resolution = resolve_target_resolution(
                params=self._params,
                resolution=resolution,
                scale=scale,
            )
            if scale is None:
                return resolution, None
            if isinstance(scale, (tuple, list)):
                scale_x = torch.as_tensor(
                    scale[0],
                    device=self._device,
                    dtype=self._dtype,
                ).reshape(())
                scale_y = torch.as_tensor(
                    scale[1],
                    device=self._device,
                    dtype=self._dtype,
                ).reshape(())
                return resolution, (scale_x, scale_y)
            scale = torch.as_tensor(scale, device=self._device, dtype=self._dtype)
            if scale.numel() == 1:
                scale = scale.reshape(())
                return resolution, (scale, scale)
            scale = scale.reshape(2)
            return resolution, (scale[0], scale[1])

        resolution, scale = _normalize_inputs(resolution=resolution, scale=scale)

        if scale is None:
            scale_x = (
                torch.as_tensor(resolution[1], dtype=self._dtype, device=self._device)
                / self._params["w"]
            )
            scale_y = (
                torch.as_tensor(resolution[0], dtype=self._dtype, device=self._device)
                / self._params["h"]
            )
        else:
            scale_x, scale_y = scale
        zero = torch.zeros((), dtype=self._dtype, device=self._device)
        one = torch.ones((), dtype=self._dtype, device=self._device)
        transform = torch.stack(
            [
                torch.stack([scale_x, zero, zero]),
                torch.stack([zero, scale_y, zero]),
                torch.stack([zero, zero, one]),
            ]
        )
        return self.transform_intrinsics(transform=transform, resolution=resolution)


class CameraIntrinsicsSimplePinhole(CameraIntrinsics):
    """Simple-pinhole intrinsics: a single shared focal length f, perspective model."""

    MODEL: ClassVar[str] = "simple_pinhole"

    @property
    def fx(self) -> torch.Tensor:
        """The shared focal length.

        Args:
            None.

        Returns:
            The scalar tensor ``params["f"]``.
        """
        return self._params["f"]

    @property
    def fy(self) -> torch.Tensor:
        """The shared focal length.

        Args:
            None.

        Returns:
            The scalar tensor ``params["f"]``.
        """
        return self._params["f"]

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

        def _validate_inputs() -> None:
            assert isinstance(points_camera, torch.Tensor), (
                "Expected points_camera to be a torch.Tensor. "
                f"{type(points_camera)=}"
            )
            assert points_camera.shape[-1] == 3, (
                "Expected points_camera last dim to be 3. " f"{points_camera.shape=}"
            )
            assert isinstance(inplace, bool), (
                "Expected inplace to be a bool. " f"{type(inplace)=}"
            )

        _validate_inputs()

        out = points_camera[..., :2] if inplace else points_camera[..., :2].clone()
        z = points_camera[..., 2]
        f, cx, cy = self.fx, self.cx, self.cy
        out[..., 0].div_(z).mul_(f).add_(cx)
        out[..., 1].div_(z).mul_(f).add_(cy)
        return out

    @property
    def fov(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The horizontal / vertical field of view in degrees.

        Args:
            None.

        Returns:
            The ``(horizontal, vertical)`` field of view in degrees as scalar tensors.
        """
        horizontal_fov = 2.0 * torch.atan(self.cx / self.fx) * 180.0 / math.pi
        vertical_fov = 2.0 * torch.atan(self.cy / self.fy) * 180.0 / math.pi
        return (horizontal_fov, vertical_fov)


class CameraIntrinsicsPinhole(CameraIntrinsics):
    """Pinhole intrinsics: independent focal lengths fx / fy, perspective model."""

    MODEL: ClassVar[str] = "pinhole"

    @property
    def fx(self) -> torch.Tensor:
        """The horizontal focal length.

        Args:
            None.

        Returns:
            The scalar tensor ``params["fx"]``.
        """
        return self._params["fx"]

    @property
    def fy(self) -> torch.Tensor:
        """The vertical focal length.

        Args:
            None.

        Returns:
            The scalar tensor ``params["fy"]``.
        """
        return self._params["fy"]

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

        def _validate_inputs() -> None:
            assert isinstance(points_camera, torch.Tensor), (
                "Expected points_camera to be a torch.Tensor. "
                f"{type(points_camera)=}"
            )
            assert points_camera.shape[-1] == 3, (
                "Expected points_camera last dim to be 3. " f"{points_camera.shape=}"
            )
            assert isinstance(inplace, bool), (
                "Expected inplace to be a bool. " f"{type(inplace)=}"
            )

        _validate_inputs()

        out = points_camera[..., :2] if inplace else points_camera[..., :2].clone()
        z = points_camera[..., 2]
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        out[..., 0].div_(z).mul_(fx).add_(cx)
        out[..., 1].div_(z).mul_(fy).add_(cy)
        return out

    @property
    def fov(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The horizontal / vertical field of view in degrees.

        Args:
            None.

        Returns:
            The ``(horizontal, vertical)`` field of view in degrees as scalar tensors.
        """
        horizontal_fov = 2.0 * torch.atan(self.cx / self.fx) * 180.0 / math.pi
        vertical_fov = 2.0 * torch.atan(self.cy / self.fy) * 180.0 / math.pi
        return (horizontal_fov, vertical_fov)


class CameraIntrinsicsOrtho(CameraIntrinsics):
    """Ortho (weak-perspective) intrinsics: focal scales fx / fy, no perspective divide."""

    MODEL: ClassVar[str] = "ortho"

    @property
    def fx(self) -> torch.Tensor:
        """The horizontal focal scale.

        Args:
            None.

        Returns:
            The scalar tensor ``params["fx"]``.
        """
        return self._params["fx"]

    @property
    def fy(self) -> torch.Tensor:
        """The vertical focal scale.

        Args:
            None.

        Returns:
            The scalar tensor ``params["fy"]``.
        """
        return self._params["fy"]

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

        def _validate_inputs() -> None:
            assert isinstance(points_camera, torch.Tensor), (
                "Expected points_camera to be a torch.Tensor. "
                f"{type(points_camera)=}"
            )
            assert points_camera.shape[-1] == 3, (
                "Expected points_camera last dim to be 3. " f"{points_camera.shape=}"
            )
            assert isinstance(inplace, bool), (
                "Expected inplace to be a bool. " f"{type(inplace)=}"
            )

        _validate_inputs()

        out = points_camera[..., :2] if inplace else points_camera[..., :2].clone()
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        out[..., 0].mul_(fx).add_(cx)
        out[..., 1].mul_(fy).add_(cy)
        return out


def build_camera_intrinsics(
    model: str,
    params: Dict[str, Union[int, float, np.ndarray, torch.Tensor]],
    intr_convention: str,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> CameraIntrinsics:
    """Build the CameraIntrinsics subclass for a camera-model string.

    The serialization-boundary factory; dispatches on the model identifier.

    Args:
        model: Camera-model identifier string.
        params: The model's named scalar intrinsics parameters; carries the resolution keys ``h`` / ``w`` alongside the projection keys.
        intr_convention: Image-plane frame the params are stated in, one of ``standard`` / ``opengl`` / ``pytorch3d`` / ``vulkan``.
        device: Optional target device for the scalar tensor params.
        dtype: Optional target floating dtype for the scalar tensor params.

    Returns:
        The CameraIntrinsics subclass instance for the model.
    """
    if model == "simple_pinhole":
        return CameraIntrinsicsSimplePinhole(
            params=params,
            intr_convention=intr_convention,
            device=device,
            dtype=dtype,
        )
    if model == "pinhole":
        return CameraIntrinsicsPinhole(
            params=params,
            intr_convention=intr_convention,
            device=device,
            dtype=dtype,
        )
    if model == "ortho":
        return CameraIntrinsicsOrtho(
            params=params,
            intr_convention=intr_convention,
            device=device,
            dtype=dtype,
        )
    assert 0, "Should not reach here. " f"{model=}"
