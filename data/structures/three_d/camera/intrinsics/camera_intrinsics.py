import math
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from data.structures.three_d.camera.intrinsics.conventions import (
    transform_intr_convention,
)
from data.structures.three_d.camera.intrinsics.validation import (
    validate_camera_intrinsics_attributes,
    validate_intr_convention,
)


class CameraIntrinsics(ABC):
    """Abstract base for a camera's intrinsics.

    Owns the named params plus device and the projection contract; each concrete subclass is exactly one camera model.
    """

    MODEL: ClassVar[str]

    def __init__(
        self,
        params: Dict[str, Union[int, float, np.ndarray, torch.Tensor]],
        intr_convention: str,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Construct a CameraIntrinsics from tensor-compatible named params.

        Args:
            params: The model's named intrinsics parameters, every one of them sharing one leading batch shape — ``[]`` for a single camera, ``[B]`` for a batch of them; carries the resolution keys ``h`` / ``w`` alongside the projection keys.
            intr_convention: Image-plane frame the params are stated in, one of ``standard`` / ``opengl`` / ``pytorch3d`` / ``vulkan``.
            device: Optional target device for the tensor params; ``None`` resolves to the device of the first tensor param, cpu when none is a tensor.
            dtype: Optional target floating dtype for the tensor params; ``None`` resolves to the dtype of the first floating tensor or numpy param, float32 when none is.

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
                # The normalization casts every param onto one floating dtype, which would turn a bool into 0 / 1 and drop an imaginary part without a word.
                if isinstance(value, np.ndarray):
                    assert np.issubdtype(value.dtype, np.number), (
                        "Expected every numpy intrinsics param to be numeric. "
                        f"{key=} {value.dtype=}"
                    )
                if isinstance(value, torch.Tensor):
                    assert not value.is_complex(), (
                        "Expected every tensor intrinsics param to be real-valued. "
                        f"{key=} {value.dtype=}"
                    )
            assert isinstance(intr_convention, str), (
                "Expected CameraIntrinsics intr_convention to be a string. "
                f"{type(intr_convention)=}"
            )
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
            device: Optional[Union[str, torch.device]],
            dtype: Optional[torch.dtype],
        ) -> Tuple[Dict[str, torch.Tensor], torch.device, torch.dtype]:
            if device is None:
                # The one exception: an unset device resolves to the given params', so a component __getitem__ rebuilds stays where its batch is.
                device = next(
                    (
                        value.device
                        for value in params.values()
                        if isinstance(value, torch.Tensor)
                    ),
                    torch.device("cpu"),
                )
            # One physical device has one spelling here, so a cuda and a cuda:0 naming it never compare unequal.
            device = torch.device(device)
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            if dtype is None:
                # The one exception: an unset dtype resolves to the given params', so a component __getitem__ rebuilds keeps the dtype its batch holds.
                dtype = next(
                    (
                        torch.as_tensor(value).dtype
                        for value in params.values()
                        if (
                            isinstance(value, torch.Tensor)
                            and value.is_floating_point()
                        )
                        or (
                            isinstance(value, np.ndarray)
                            and np.issubdtype(value.dtype, np.floating)
                        )
                    ),
                    torch.float32,
                )
            # Every param follows the resolved device and dtype, never the other way around.
            params = {
                key: torch.as_tensor(value, device=device, dtype=dtype)
                for key, value in params.items()
            }
            return params, device, dtype

        params, device, dtype = _normalize_inputs(
            params=params,
            device=device,
            dtype=dtype,
        )

        validate_camera_intrinsics_attributes(
            model=type(self).MODEL,
            intr_convention=intr_convention,
            params=params,
            device=device,
            dtype=dtype,
        )

        self._params: Dict[str, torch.Tensor] = params
        self._intr_convention: str = intr_convention
        self._device: torch.device = device
        self._dtype: torch.dtype = dtype

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
            The image-plane convention string (standard / opengl / pytorch3d / vulkan), without which a principal point names no location.
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

    def __getitem__(
        self, index: Union[int, slice, List[int], None]
    ) -> "CameraIntrinsics":
        """Index the leading batch axis the params carry.

        Args:
            index: The index applied to every param's leading axis the way the tensors index their own, so ``None`` adds an axis of one and an int drops it.

        Returns:
            A CameraIntrinsics of the same model whose params carry the indexed leading axis.
        """
        # a pass over the model's few param names, never over the cameras
        params = {key: value[index] for key, value in self._params.items()}
        return build_camera_intrinsics(
            model=type(self).MODEL,
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
                Tuple[Union[int, float], Union[int, float]],
                List[Union[int, float]],
                np.ndarray,
                torch.Tensor,
            ]
        ] = None,
    ) -> "CameraIntrinsics":
        """Return this CameraIntrinsics restated against a different resolution.

        The diagonal case of an intrinsics transform, so this builds that transform and the one owner applies it. Exactly one of ``resolution`` or ``scale`` must be provided.

        Args:
            resolution: Optional target image resolution as one integer side or ``(height, width)``.
            scale: Optional uniform scale, or a per-axis ``(sx, sy)`` pair.

        Returns:
            A new CameraIntrinsics of the same model stated against the target resolution.
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
                    Tuple[Union[int, float], Union[int, float]],
                    List[Union[int, float]],
                    np.ndarray,
                    torch.Tensor,
                ]
            ],
        ) -> Tuple[
            Tuple[Union[int, torch.Tensor], Union[int, torch.Tensor]],
            torch.Tensor,
            torch.Tensor,
        ]:
            resolution = _resolve_target_resolution(
                params=self._params,
                resolution=resolution,
                scale=scale,
            )
            if scale is not None:
                # Taken raw rather than re-derived from resolution, which _resolve_target_resolution detached and rounded to whole pixels, severing a tensor factor from the autograd graph.
                if isinstance(scale, (tuple, list)):
                    sx = torch.as_tensor(
                        scale[0],
                        device=self._device,
                        dtype=self._dtype,
                    ).reshape(())
                    sy = torch.as_tensor(
                        scale[1],
                        device=self._device,
                        dtype=self._dtype,
                    ).reshape(())
                else:
                    scale = torch.as_tensor(
                        scale, device=self._device, dtype=self._dtype
                    )
                    if scale.numel() == 1:
                        scale = scale.reshape(())
                        sx, sy = scale, scale
                    else:
                        scale = scale.reshape(2)
                        sx, sy = scale[0], scale[1]
            else:
                # The size the params are already stated against is two of those params, the one place every model states it.
                sx = (
                    torch.as_tensor(
                        resolution[1], dtype=self._dtype, device=self._device
                    )
                    / self._params["w"]
                )
                sy = (
                    torch.as_tensor(
                        resolution[0], dtype=self._dtype, device=self._device
                    )
                    / self._params["h"]
                )
            return resolution, sx, sy

        resolution, sx, sy = _normalize_inputs(resolution=resolution, scale=scale)

        # A rounded raster and a raw factor are not exactly consistent when the product is not whole; the gradient is what this trade keeps.
        zero = torch.zeros_like(sx)
        one = torch.ones_like(sx)
        transform = torch.stack(
            [
                torch.stack([sx, zero, zero], dim=-1),
                torch.stack([zero, sy, zero], dim=-1),
                torch.stack([zero, zero, one], dim=-1),
            ],
            dim=-2,
        )
        return self.transform_intrinsics(transform=transform, resolution=resolution)

    def transform_intrinsics(
        self,
        transform: torch.Tensor,
        resolution: Tuple[Union[int, torch.Tensor], Union[int, torch.Tensor]],
    ) -> "CameraIntrinsics":
        """Return this CameraIntrinsics restated onto another image by a pixel-frame affine.

        The raster that image is named alongside it, because a 3x3 carries no size of its own.

        Args:
            transform: Axis-aligned pixel-frame affine as a ``(..., 3, 3)`` floating torch.Tensor whose ``[..., 0, 1]`` / ``[..., 1, 0]`` entries are zero and whose last row is ``[0, 0, 1]``, the leading dims being the camera batch a single affine leaves empty.
            resolution: The target image's own resolution as ``(height, width)`` integer values, scalar integer-valued tensors, or ``[B]`` integer-valued tensors naming one side per camera.

        Returns:
            A new CameraIntrinsics of the same model, on this intrinsics' own image-plane frame, stated against ``resolution``.
        """

        def _validate_inputs() -> None:
            assert isinstance(transform, torch.Tensor), (
                "Expected the intrinsics transform to be a torch.Tensor. "
                f"{type(transform)=}"
            )
            assert transform.shape[-2:] == (3, 3), (
                "Expected the intrinsics transform trailing dims to be (3, 3). "
                f"{transform.shape=}"
            )
            assert transform.is_floating_point(), (
                "Expected the intrinsics transform dtype to be floating. "
                f"{transform.dtype=}"
            )
            last_row = transform[..., 2, :].detach().cpu()
            assert torch.equal(
                last_row,
                torch.tensor([0.0, 0.0, 1.0], dtype=transform.dtype).expand(
                    last_row.shape
                ),
            ), (
                "Expected the intrinsics transform last row to be [0, 0, 1]. "
                f"{last_row=}"
            )
            # An axis-aligned affine is the only kind that keeps a skew-free K skew-free.
            assert bool(torch.all(transform[..., 0, 1] == 0.0)) and bool(
                torch.all(transform[..., 1, 0] == 0.0)
            ), (
                "Expected the intrinsics transform to be axis-aligned, its "
                "off-diagonal entries [..., 0, 1] and [..., 1, 0] zero. "
                f"{transform[..., 0, 1]=} {transform[..., 1, 0]=}"
            )
            assert isinstance(resolution, tuple) and len(resolution) == 2, (
                "Expected resolution to be a (height, width) tuple of length 2. "
                f"{resolution=}"
            )
            for value in resolution:
                assert isinstance(value, (int, torch.Tensor)), (
                    "Expected resolution values to be integers or tensors. "
                    f"{type(value)=} {resolution=}"
                )
                if isinstance(value, torch.Tensor):
                    assert value.ndim <= 1, (
                        "Expected tensor resolution values to be scalar or to carry "
                        f"one entry per camera. {value.shape=}"
                    )
                    assert torch.equal(value, torch.round(value)), (
                        "Expected tensor resolution values to be integer-valued. "
                        f"{value=}"
                    )
                    assert bool(torch.all(value > 0)), (
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
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            transform = transform.to(device=self._device, dtype=self._dtype)
            resolution = tuple(
                torch.as_tensor(value, device=self._device, dtype=self._dtype)
                for value in resolution
            )
            return transform, resolution

        transform, resolution = _normalize_inputs(
            transform=transform,
            resolution=resolution,
        )

        # An affine between two rasters composes only with a K stated in them, so the camera is read in pixels.
        standard = self.to(intr_convention="standard")
        zero = torch.zeros_like(standard.fx)
        one = torch.ones_like(standard.fx)
        K = transform @ torch.stack(
            [
                torch.stack([standard.fx, zero, standard.cx], dim=-1),
                torch.stack([zero, standard.fy, standard.cy], dim=-1),
                torch.stack([zero, zero, one], dim=-1),
            ],
            dim=-2,
        )
        params = self._focal_params(fx=K[..., 0, 0], fy=K[..., 1, 1])
        params["cx"] = K[..., 0, 2]
        params["cy"] = K[..., 1, 2]
        # A single raster names the same sides for every camera of a batch.
        params["h"] = torch.broadcast_to(resolution[0], K.shape[:-2])
        params["w"] = torch.broadcast_to(resolution[1], K.shape[:-2])
        transformed = type(self)(params=params, intr_convention="standard")
        intrinsics = transformed.to(intr_convention=self._intr_convention)
        return intrinsics

    @property
    def cx(self) -> torch.Tensor:
        """The horizontal principal-point coordinate.

        Args:
            None.

        Returns:
            The ``[]`` (unbatched) or ``[B]`` (batched) tensor ``params["cx"]``.
        """
        return self._params["cx"]

    @property
    def cy(self) -> torch.Tensor:
        """The vertical principal-point coordinate.

        Args:
            None.

        Returns:
            The ``[]`` (unbatched) or ``[B]`` (batched) tensor ``params["cy"]``.
        """
        return self._params["cy"]

    @property
    def resolution(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The resolution these params are stated against.

        Args:
            None.

        Returns:
            The ``(height, width)`` tensor pair, each ``[]`` or ``[B]``, read off the two params that carry it, since a principal point in pixels names a location only against them.
        """
        return self._params["h"], self._params["w"]

    @property
    @abstractmethod
    def fx(self) -> torch.Tensor:
        """The horizontal focal length / scale, whose params key differs per model.

        Args:
            None.

        Returns:
            The horizontal focal length / scale as a ``[]`` or ``[B]`` tensor.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def fy(self) -> torch.Tensor:
        """The vertical focal length / scale, whose params key differs per model.

        Args:
            None.

        Returns:
            The vertical focal length / scale as a ``[]`` or ``[B]`` tensor.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _focal_params(
        cls, fx: torch.Tensor, fy: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """The inverse of the fx / fy accessors: state a horizontal and a vertical focal in this model's own focal params.

        Args:
            fx: The horizontal focal length / scale as a ``[]`` or ``[B]`` tensor.
            fy: The vertical focal length / scale, shaped like ``fx``.

        Returns:
            This model's focal params keyed by their own names.
        """
        raise NotImplementedError

    @abstractmethod
    def project(
        self, points_camera: torch.Tensor, inplace: bool = False
    ) -> torch.Tensor:
        """Map camera-space 3D points to 2D image points under this model.

        Each param is unsqueezed against the point axis, so a ``[B]`` param batch projects a ``[B, N, 3]`` point batch in one op.

        Args:
            points_camera: Camera-space points, a ``[..., 3]`` torch.Tensor whose leading axes carry the params' own batch shape.
            inplace: If True, project in place — write the image points over the first two columns of ``points_camera`` and return a ``[..., 2]`` view aliasing that input (its depth column is left intact). If False, return a freshly allocated ``[..., 2]`` and leave ``points_camera`` unchanged.

        Returns:
            The ``[..., 2]`` image points torch.Tensor (a view into ``points_camera`` when inplace, else a new tensor).
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

        The intrinsics half of the frame change its extrinsics counterpart performs on the pose.

        Args:
            device: Target device; ``None`` keeps the current device.
            dtype: Target floating dtype; ``None`` keeps the current dtype.
            non_blocking: Whether tensor moves may be asynchronous.
            copy: Whether tensor moves must allocate new storage even when unchanged.
            intr_convention: Target image-plane frame; ``None`` keeps the current one.

        Returns:
            This CameraIntrinsics when unchanged, else a CameraIntrinsics of the same model on the target device, dtype, and image-plane frame.
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
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
                copy=copy,
            )
            for key, value in params.items()
        }
        if (
            (device is None or torch.device(device) == self._device)
            and (dtype is None or dtype == self._dtype)
            and (intr_convention is None or intr_convention == self._intr_convention)
            and copy is False
        ):
            return self
        intrinsics = type(self)(
            params=params,
            intr_convention=intr_convention or self._intr_convention,
        )
        return intrinsics


class CameraIntrinsicsSimplePinhole(CameraIntrinsics):
    """Simple-pinhole intrinsics: a single shared focal length f, perspective model."""

    MODEL: ClassVar[str] = "simple_pinhole"

    @property
    def fx(self) -> torch.Tensor:
        """The shared focal length.

        Args:
            None.

        Returns:
            The ``[]`` (unbatched) or ``[B]`` (batched) tensor ``params["f"]``.
        """
        return self._params["f"]

    @property
    def fy(self) -> torch.Tensor:
        """The shared focal length.

        Args:
            None.

        Returns:
            The ``[]`` (unbatched) or ``[B]`` (batched) tensor ``params["f"]``.
        """
        return self._params["f"]

    @classmethod
    def _focal_params(
        cls, fx: torch.Tensor, fy: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """State the pair as the one shared focal f, since this model states its two focals as one f.

        Args:
            fx: The horizontal focal length as a ``[]`` or ``[B]`` tensor.
            fy: The vertical focal length, shaped like ``fx``.

        Returns:
            The ``{"f": fx}`` focal params.
        """
        # One shared f holds one ratio, so an affine scaling the axes apart leaves this model nothing to state the second in.
        assert bool(torch.all(torch.isclose(fx, fy))), (
            "Expected the horizontal and vertical focal to agree at every entry for "
            "simple_pinhole, whose one shared f holds one ratio. "
            f"{fx=} {fy=}"
        )
        return {"f": fx}

    def project(
        self, points_camera: torch.Tensor, inplace: bool = False
    ) -> torch.Tensor:
        """Perspective projection with a single shared focal length.

        Args:
            points_camera: Camera-space points, a ``[..., 3]`` torch.Tensor.
            inplace: If True, project in place — write the image points over the first two columns of ``points_camera`` and return a ``[..., 2]`` view aliasing that input (its depth column is left intact). If False, return a freshly allocated ``[..., 2]`` and leave ``points_camera`` unchanged.

        Returns:
            The ``[..., 2]`` image points torch.Tensor (a view into ``points_camera`` when inplace, else a new tensor).
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
        # each param unsqueezed against the point axis so a [B] param batch aligns with [B, N] points
        f, cx, cy = self.fx[..., None], self.cx[..., None], self.cy[..., None]
        out[..., 0].div_(z).mul_(f).add_(cx)
        out[..., 1].div_(z).mul_(f).add_(cy)
        return out

    @property
    def fov(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The horizontal / vertical field of view in degrees.

        Args:
            None.

        Returns:
            The ``(horizontal, vertical)`` field of view in degrees, each a ``[]`` or ``[B]`` tensor.
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
            The ``[]`` (unbatched) or ``[B]`` (batched) tensor ``params["fx"]``.
        """
        return self._params["fx"]

    @property
    def fy(self) -> torch.Tensor:
        """The vertical focal length.

        Args:
            None.

        Returns:
            The ``[]`` (unbatched) or ``[B]`` (batched) tensor ``params["fy"]``.
        """
        return self._params["fy"]

    @classmethod
    def _focal_params(
        cls, fx: torch.Tensor, fy: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """State the pair as this model's independent fx / fy focal lengths.

        Args:
            fx: The horizontal focal length as a ``[]`` or ``[B]`` tensor.
            fy: The vertical focal length, shaped like ``fx``.

        Returns:
            The ``{"fx": fx, "fy": fy}`` focal params.
        """
        return {"fx": fx, "fy": fy}

    def project(
        self, points_camera: torch.Tensor, inplace: bool = False
    ) -> torch.Tensor:
        """Perspective projection with independent fx / fy.

        Args:
            points_camera: Camera-space points, a ``[..., 3]`` torch.Tensor.
            inplace: If True, project in place — write the image points over the first two columns of ``points_camera`` and return a ``[..., 2]`` view aliasing that input (its depth column is left intact). If False, return a freshly allocated ``[..., 2]`` and leave ``points_camera`` unchanged.

        Returns:
            The ``[..., 2]`` image points torch.Tensor (a view into ``points_camera`` when inplace, else a new tensor).
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
        # each param unsqueezed against the point axis so a [B] param batch aligns with [B, N] points
        fx, fy, cx, cy = (
            self.fx[..., None],
            self.fy[..., None],
            self.cx[..., None],
            self.cy[..., None],
        )
        out[..., 0].div_(z).mul_(fx).add_(cx)
        out[..., 1].div_(z).mul_(fy).add_(cy)
        return out

    @property
    def fov(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The horizontal / vertical field of view in degrees.

        Args:
            None.

        Returns:
            The ``(horizontal, vertical)`` field of view in degrees, each a ``[]`` or ``[B]`` tensor.
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
            The ``[]`` (unbatched) or ``[B]`` (batched) tensor ``params["fx"]``.
        """
        return self._params["fx"]

    @property
    def fy(self) -> torch.Tensor:
        """The vertical focal scale.

        Args:
            None.

        Returns:
            The ``[]`` (unbatched) or ``[B]`` (batched) tensor ``params["fy"]``.
        """
        return self._params["fy"]

    @classmethod
    def _focal_params(
        cls, fx: torch.Tensor, fy: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """State the pair as this model's independent fx / fy focal scales.

        Args:
            fx: The horizontal focal scale as a ``[]`` or ``[B]`` tensor.
            fy: The vertical focal scale, shaped like ``fx``.

        Returns:
            The ``{"fx": fx, "fy": fy}`` focal params.
        """
        return {"fx": fx, "fy": fy}

    def project(
        self, points_camera: torch.Tensor, inplace: bool = False
    ) -> torch.Tensor:
        """Orthographic projection: scale and offset without the perspective divide.

        Args:
            points_camera: Camera-space points, a ``[..., 3]`` torch.Tensor.
            inplace: If True, project in place — write the image points over the first two columns of ``points_camera`` and return a ``[..., 2]`` view aliasing that input (its depth column is left intact). If False, return a freshly allocated ``[..., 2]`` and leave ``points_camera`` unchanged.

        Returns:
            The ``[..., 2]`` image points torch.Tensor (a view into ``points_camera`` when inplace, else a new tensor).
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
        # each param unsqueezed against the point axis so a [B] param batch aligns with [B, N] points
        fx, fy, cx, cy = (
            self.fx[..., None],
            self.fy[..., None],
            self.cx[..., None],
            self.cy[..., None],
        )
        out[..., 0].mul_(fx).add_(cx)
        out[..., 1].mul_(fy).add_(cy)
        return out


def _resolve_target_resolution(
    params: Dict[str, torch.Tensor],
    resolution: Optional[
        Union[int, Tuple[int, int], List[int], np.ndarray, torch.Tensor]
    ] = None,
    scale: Optional[
        Union[
            int,
            float,
            Tuple[Union[int, float], Union[int, float]],
            List[Union[int, float]],
            np.ndarray,
            torch.Tensor,
        ]
    ] = None,
) -> Tuple[Union[int, torch.Tensor], Union[int, torch.Tensor]]:
    """Resolve the two ways a caller names a target resolution into the single form a rescale reads.

    Args:
        params: The model's named intrinsics params, carrying the ``h`` / ``w`` values the current resolution is read off, scalar for one camera and ``[B]`` for a batch.
        resolution: Optional target image resolution as one integer side or ``(height, width)``.
        scale: Optional uniform factor, or a per-axis ``(sx, sy)`` pair, on the resolution the params already carry.

    Returns:
        The target image resolution as a ``(height, width)`` pair: positive ints when the resolution is given, and int64 torch.Tensors shaped like the params' ``h`` / ``w`` (``[]`` for one camera, ``[B]`` for a batch, one side per camera) when a factor is applied.
    """

    def _validate_inputs() -> None:
        assert (resolution is None) ^ (scale is None), (
            "Expected exactly one of resolution or scale to be provided. "
            f"{resolution=} {scale=}"
        )
        if resolution is not None:
            assert isinstance(
                resolution, (int, tuple, list, np.ndarray, torch.Tensor)
            ), (
                "Expected resolution to be a positive int or length-2 array-like. "
                f"{type(resolution)=}"
            )
            if isinstance(resolution, int):
                assert resolution > 0, (
                    "Expected scalar resolution to be positive. " f"{resolution=}"
                )
            elif isinstance(resolution, (tuple, list)):
                assert len(resolution) == 2, (
                    "Expected resolution to have length 2. " f"{resolution=}"
                )
                assert all(isinstance(item, int) for item in resolution), (
                    "Expected resolution values to be integers. " f"{resolution=}"
                )
                assert all(item > 0 for item in resolution), (
                    "Expected resolution values to be positive. " f"{resolution=}"
                )
            elif isinstance(resolution, np.ndarray):
                assert resolution.size in (1, 2), (
                    "Expected numpy resolution to contain one or two values. "
                    f"{resolution.shape=}"
                )
                assert np.issubdtype(resolution.dtype, np.integer), (
                    "Expected numpy resolution values to be integers. "
                    f"{resolution.dtype=}"
                )
                assert bool(np.all(resolution > 0)), (
                    "Expected numpy resolution values to be positive. " f"{resolution=}"
                )
            elif isinstance(resolution, torch.Tensor):
                assert resolution.numel() in (1, 2), (
                    "Expected tensor resolution to contain one or two values. "
                    f"{resolution.shape=}"
                )
                assert not resolution.is_floating_point(), (
                    "Expected tensor resolution values to be integers. "
                    f"{resolution.dtype=}"
                )
                assert bool(torch.all(resolution > 0)), (
                    "Expected tensor resolution values to be positive. "
                    f"{resolution=}"
                )
        if scale is not None:
            assert isinstance(
                scale, (int, float, tuple, list, np.ndarray, torch.Tensor)
            ), (
                "Expected scale to be a positive number or length-2 array-like. "
                f"{type(scale)=}"
            )
            if isinstance(scale, (int, float)):
                assert float(scale) > 0.0, (
                    "Expected scalar scale to be positive. " f"{scale=}"
                )
            elif isinstance(scale, (tuple, list)):
                assert len(scale) == 2, "Expected scale to have length 2. " f"{scale=}"
                assert all(
                    isinstance(item, (int, float, torch.Tensor)) for item in scale
                ), (
                    "Expected scale values to be numbers or scalar tensors. "
                    f"{scale=}"
                )
            elif isinstance(scale, np.ndarray):
                assert scale.size in (1, 2), (
                    "Expected numpy scale to contain one or two values. "
                    f"{scale.shape=}"
                )
                assert np.issubdtype(scale.dtype, np.number), (
                    "Expected numpy scale values to be numeric. " f"{scale.dtype=}"
                )
                assert bool(np.all(scale > 0)), (
                    "Expected numpy scale values to be positive. " f"{scale=}"
                )
            elif isinstance(scale, torch.Tensor):
                assert scale.numel() in (1, 2), (
                    "Expected tensor scale to contain one or two values. "
                    f"{scale.shape=}"
                )
                assert scale.is_floating_point(), (
                    "Expected tensor scale values to be floating. " f"{scale.dtype=}"
                )
                assert bool(torch.all(scale > 0)), (
                    "Expected tensor scale values to be positive. " f"{scale=}"
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
                Tuple[Union[int, float, torch.Tensor], Union[int, float, torch.Tensor]],
                List[Union[int, float, torch.Tensor]],
                np.ndarray,
                torch.Tensor,
            ]
        ],
    ) -> Tuple[
        Optional[Tuple[int, int]],
        Optional[
            Tuple[Union[int, float, torch.Tensor], Union[int, float, torch.Tensor]]
        ],
    ]:
        if resolution is not None:
            if isinstance(resolution, int):
                resolution = (resolution, resolution)
            elif isinstance(resolution, (tuple, list)):
                resolution = (int(resolution[0]), int(resolution[1]))
            elif isinstance(resolution, np.ndarray):
                values = resolution.reshape(-1)
                if values.size == 1:
                    resolution = (int(values[0]), int(values[0]))
                else:
                    resolution = (int(values[0]), int(values[1]))
            elif isinstance(resolution, torch.Tensor):
                values = resolution.reshape(-1)
                if values.numel() == 1:
                    side = int(values[0].detach().cpu().item())
                    resolution = (side, side)
                else:
                    resolution = (
                        int(values[0].detach().cpu().item()),
                        int(values[1].detach().cpu().item()),
                    )
        if scale is not None:
            if isinstance(scale, (int, float)):
                scale = (scale, scale)
            elif isinstance(scale, np.ndarray):
                values = scale.reshape(-1)
                if values.size == 1:
                    scale = (float(values[0]), float(values[0]))
                else:
                    scale = (float(values[0]), float(values[1]))
            elif isinstance(scale, torch.Tensor):
                values = scale.reshape(-1)
                if values.numel() == 1:
                    scale = (values[0], values[0])
                else:
                    scale = (values[0], values[1])
            elif isinstance(scale, list):
                scale = (scale[0], scale[1])
        return resolution, scale

    resolution, scale = _normalize_inputs(resolution=resolution, scale=scale)

    if resolution is not None:
        return resolution
    if scale is not None:
        height = torch.round(
            torch.as_tensor(params["h"]).detach().cpu().double()
            * torch.as_tensor(scale[1]).detach().cpu().double()
        ).long()
        width = torch.round(
            torch.as_tensor(params["w"]).detach().cpu().double()
            * torch.as_tensor(scale[0]).detach().cpu().double()
        ).long()
        assert bool(torch.all(height > 0)) and bool(torch.all(width > 0)), (
            "Expected a scale that keeps both image sides positive. "
            f"{height=} {width=} {scale=}"
        )
        return height, width
    assert 0, "Should not reach here. " f"{resolution=} {scale=}"


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
        params: The model's named intrinsics parameters, every one of them sharing one leading batch shape — ``[]`` for a single camera, ``[B]`` for a batch of them; carries the resolution keys ``h`` / ``w`` alongside the projection keys.
        intr_convention: Image-plane frame the params are stated in, one of ``standard`` / ``opengl`` / ``pytorch3d`` / ``vulkan``.
        device: Optional target device for the tensor params.
        dtype: Optional target floating dtype for the tensor params.

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
