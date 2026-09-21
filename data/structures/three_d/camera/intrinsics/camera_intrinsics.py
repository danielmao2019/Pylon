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
from utils.ops.apply import apply_tensor_op


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
            device: Optional target device for the tensor params; ``None`` resolves to the one device the tensor params share, cpu when none is a tensor.
            dtype: Optional target floating dtype for the tensor params; ``None`` resolves to the one dtype the floating tensor or numpy params share, float32 when none is.

        Returns:
            None.
        """

        def _validate_inputs() -> None:
            validate_camera_intrinsics_attributes(
                model=type(self).MODEL,
                intr_convention=intr_convention,
                params=params,
                device=device,
                dtype=dtype,
            )

        _validate_inputs()

        def _normalize_inputs(
            params: Dict[str, Union[int, float, np.ndarray, torch.Tensor]],
            device: Optional[Union[str, torch.device]],
            dtype: Optional[torch.dtype],
        ) -> Tuple[Dict[str, torch.Tensor], torch.device, torch.dtype]:
            if device is None:
                # A set of every tensor param's device, so no param is the one read.
                param_devices = {
                    value.device
                    for value in params.values()
                    if isinstance(value, torch.Tensor)
                }
                if len(param_devices) > 0:
                    # Single, since validate_camera_intrinsics_attributes asserts the tensor params share one; the one exception: an unset device resolves to the given params', so a component __getitem__ rebuilds stays where its batch is.
                    (device,) = param_devices
                else:
                    device = torch.device("cpu")
            device = torch.device(device)
            # One physical device has one spelling here, so a cuda and a cuda:0 naming it never compare unequal.
            if device.type == "cuda" and device.index is None:
                # Where a tensor sent to a bare cuda lands, and so the device it reports.
                device = torch.device("cuda", torch.cuda.current_device())
            if dtype is None:
                # A set of every floating param's dtype, so no param is the one read.
                param_dtypes = {
                    torch.as_tensor(value).dtype
                    for value in params.values()
                    if (isinstance(value, torch.Tensor) and value.is_floating_point())
                    or (
                        isinstance(value, np.ndarray)
                        and np.issubdtype(value.dtype, np.floating)
                    )
                }
                if len(param_dtypes) > 0:
                    # Single, since validate_camera_intrinsics_attributes asserts the floating params share one; the one exception: an unset dtype resolves to the given params', so a component __getitem__ rebuilds keeps the dtype its batch holds.
                    (dtype,) = param_dtypes
                else:
                    dtype = torch.float32
            # Every param follows the resolved device and dtype, never the other way around.
            materialized_params = {}
            for key, value in params.items():
                materialized_params[key] = torch.as_tensor(
                    value, device=device, dtype=dtype
                )
            return materialized_params, device, dtype

        params, device, dtype = _normalize_inputs(
            params=params,
            device=device,
            dtype=dtype,
        )

        # A set of every param's own shape, so no param is the one read.
        param_shapes = {value.shape for value in params.values()}
        # Single, since validate_camera_intrinsics_params asserts the params share one shape.
        (batch_shape,) = param_shapes
        # None where the params are scalars: an unbatched intrinsics carries no batch axis, and states one camera.
        batch_size = batch_shape[0] if batch_shape != () else None
        self._params: Dict[str, torch.Tensor] = params
        self._intr_convention: str = intr_convention
        # The resolved device the params were built on, not read back off them.
        self._device: torch.device = device
        # The resolved dtype the params were built in, not read back off them.
        self._dtype: torch.dtype = dtype
        self._batch_size: Optional[int] = batch_size

    @property
    def model(self) -> str:
        """The camera-model identifier.

        Args:
            None.

        Returns:
            The model identifier ``type(self).MODEL``.
        """
        model = type(self).MODEL
        return model

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

    @property
    def is_batched(self) -> bool:
        """Whether the params carry a batch axis.

        Args:
            None.

        Returns:
            True for ``[B]`` params, False for the scalar params of the one camera an unbatched intrinsics states.
        """
        return self._batch_size is not None

    def __len__(self) -> int:
        """The extent of the batch axis these intrinsics carry.

        Args:
            None.

        Returns:
            The number ``B`` of cameras the ``[B]`` params of a batched intrinsics carry; an unbatched intrinsics has no length.
        """
        # Scalar params carry no batch axis, so they have no length.
        assert self._batch_size is not None, (
            "Expected a batched CameraIntrinsics, since scalar params carry no batch "
            f"axis and so have no length. {self._params=}"
        )
        return self._batch_size

    def __getitem__(
        self, index: Union[int, slice, List[int], None]
    ) -> "CameraIntrinsics":
        """Index the leading batch axis the params carry.

        Args:
            index: The index applied to every param's leading axis the way the tensors index their own, so ``None`` adds an axis of one and an int drops it.

        Returns:
            A CameraIntrinsics of the same model whose params carry the indexed leading axis.
        """
        params = {}
        # A pass over the model's few param names, never over the cameras.
        for key, value in self._params.items():
            params[key] = value[index]
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
            # A target resolution and a factor are two ways to name the same thing, and giving both leaves unstated which one wins.
            assert (resolution is None) ^ (scale is None), (
                "Expected exactly one of resolution or scale to be provided. "
                f"{resolution=} {scale=}"
            )
            if resolution is not None:
                assert (isinstance(resolution, int) and resolution > 0) or (
                    (
                        isinstance(resolution, (tuple, list))
                        or (
                            isinstance(resolution, (np.ndarray, torch.Tensor))
                            and resolution.ndim == 1
                        )
                    )
                    and len(resolution) == 2
                ), (
                    "Expected resolution to be a positive int or a length-2 "
                    f"array-like. {resolution=}"
                )
                if isinstance(resolution, (tuple, list, np.ndarray, torch.Tensor)):
                    assert (
                        isinstance(resolution[0], (int, float, np.number, torch.Tensor))
                        and isinstance(
                            resolution[1], (int, float, np.number, torch.Tensor)
                        )
                        and float(resolution[0]) > 0
                        and float(resolution[0]).is_integer()
                        and float(resolution[1]) > 0
                        and float(resolution[1]).is_integer()
                    ), (
                        "Expected both resolution entries to be positive "
                        f"integer-valued numbers. {resolution=}"
                    )
                return
            if scale is not None:
                assert (
                    (isinstance(scale, (int, float)) and scale > 0)
                    or (
                        isinstance(scale, (tuple, list))
                        and len(scale) == 2
                        and isinstance(scale[0], (int, float, np.number, torch.Tensor))
                        and isinstance(scale[1], (int, float, np.number, torch.Tensor))
                        and float(scale[0]) > 0
                        and float(scale[1]) > 0
                    )
                    or (
                        isinstance(scale, (np.ndarray, torch.Tensor))
                        and torch.as_tensor(scale).numel() in (1, 2)
                        and bool(torch.all(torch.as_tensor(scale) > 0))
                    )
                ), (
                    "Expected scale to be a positive int or float, a length-2 tuple "
                    "or list of positive numbers, or a numpy array or torch.Tensor "
                    f"holding one or two positive numbers. {scale=}"
                )
                return
            assert 0, "Should not reach here."

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
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        ]:
            def _normalize_resolution() -> Tuple[int, int]:
                """Restate the given resolution as the (h, w) pair of ints a rescale reads.

                Args:
                    None; reads the enclosing call's validated ``resolution``, one positive int side or a length-2 array-like ``(height, width)``.

                Returns:
                    The target resolution as a ``(height, width)`` pair of Python ints.
                """
                if isinstance(resolution, int):
                    return (resolution, resolution)
                if isinstance(resolution, (tuple, list, np.ndarray, torch.Tensor)):
                    return (int(resolution[0]), int(resolution[1]))
                assert 0, "Should not reach here."

            def _normalize_scale() -> torch.Tensor:
                """Restate the given scale as the [2] (sx, sy) tensor a rescale reads, on self._device in self._dtype.

                Args:
                    None; reads the enclosing call's validated ``scale``, one factor or an ``(sx, sy)`` pair.

                Returns:
                    The ``[2]`` ``(sx, sy)`` torch.Tensor on this intrinsics' device and dtype, still on the autograd graph of a tensor factor.
                """
                if isinstance(scale, (int, float)):
                    # One factor names the same one on both axes, in the (sx, sy) form the pair case arrives in.
                    return torch.tensor(
                        [scale, scale], device=self._device, dtype=self._dtype
                    )
                if isinstance(scale, (tuple, list)):
                    return torch.stack(
                        [
                            torch.as_tensor(
                                scale[0], device=self._device, dtype=self._dtype
                            ).reshape(()),
                            torch.as_tensor(
                                scale[1], device=self._device, dtype=self._dtype
                            ).reshape(()),
                        ]
                    )
                if (
                    isinstance(scale, (np.ndarray, torch.Tensor))
                    and torch.as_tensor(scale).numel() == 1
                ):
                    # One factor names the same one on both axes.
                    return (
                        torch.as_tensor(scale, device=self._device, dtype=self._dtype)
                        .reshape(())
                        .repeat(2)
                    )
                if (
                    isinstance(scale, (np.ndarray, torch.Tensor))
                    and torch.as_tensor(scale).numel() == 2
                ):
                    return torch.as_tensor(
                        scale, device=self._device, dtype=self._dtype
                    ).reshape(2)
                assert 0, "Should not reach here."

            # The target named by its size, so the factor is that size over the one the params already state.
            if resolution is not None:
                resolution = _normalize_resolution()
                # The size the params are already stated against is two of those params, the one place every model states it.
                scale = (
                    resolution[1] / self._params["w"],
                    resolution[0] / self._params["h"],
                )
                return resolution, scale
            # The target named by a factor, so the size is that factor on the one the params already state.
            if scale is not None:
                # Taken raw rather than re-derived from the resolution below, which is detached and rounded to whole pixels, severing a tensor factor from the autograd graph.
                scale = _normalize_scale()
                height = torch.round(
                    self._params["h"].detach().cpu().double()
                    * scale[1].detach().cpu().double()
                ).long()
                width = torch.round(
                    self._params["w"].detach().cpu().double()
                    * scale[0].detach().cpu().double()
                ).long()
                # A factor small enough to round a side to zero names no image, which the rebuilt intrinsics' own validation refuses.
                resolution = (height, width)
                return resolution, scale
            assert 0, "Should not reach here."

        resolution, scale = _normalize_inputs(resolution=resolution, scale=scale)

        # A rounded raster and a raw factor are not exactly consistent when the product is not whole; the gradient is what this trade keeps.
        # A resize scales both axes about the pixel frame's own origin, its top-left corner, which is what makes it diagonal.
        transform = torch.stack(
            [
                torch.stack(
                    [scale[0], torch.zeros_like(scale[0]), torch.zeros_like(scale[0])],
                    dim=-1,
                ),
                torch.stack(
                    [torch.zeros_like(scale[0]), scale[1], torch.zeros_like(scale[0])],
                    dim=-1,
                ),
                torch.stack(
                    [
                        torch.zeros_like(scale[0]),
                        torch.zeros_like(scale[0]),
                        torch.ones_like(scale[0]),
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )
        intrinsics = self.transform_intrinsics(
            transform=transform, resolution=resolution
        )
        return intrinsics

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
            # One affine, or one per camera of a batch.
            assert transform.ndim in (2, 3) and transform.shape[-2:] == (3, 3), (
                "Expected the intrinsics transform to be one (3, 3) affine or a "
                f"(T, 3, 3) stack of them. {transform.shape=}"
            )
            # One affine for every camera, or one per camera of this batch.
            assert (
                transform.ndim == 2
                or not self.is_batched
                or transform.shape[0] == len(self)
            ), (
                "Expected a stacked intrinsics transform to carry one affine per "
                "camera of this batch. "
                f"{transform.shape=} {self.is_batched=} {self._batch_size=}"
            )
            assert transform.is_floating_point(), (
                "Expected the intrinsics transform dtype to be floating. "
                f"{transform.dtype=}"
            )
            # An affine's last row.
            assert bool(
                torch.all(
                    transform[..., 2, :]
                    == torch.tensor(
                        [0.0, 0.0, 1.0],
                        device=transform.device,
                        dtype=transform.dtype,
                    )
                )
            ), (
                "Expected the intrinsics transform last row to be [0, 0, 1]. "
                f"{transform[..., 2, :]=}"
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
            # A batch scales each camera's own raster, so a side may differ per camera; positive sides of at most one axis are the rebuilt intrinsics' own validation of its h and w.
            assert (
                isinstance(resolution[0], int)
                or (
                    isinstance(resolution[0], torch.Tensor)
                    and torch.equal(resolution[0], torch.round(resolution[0]))
                )
            ) and (
                isinstance(resolution[1], int)
                or (
                    isinstance(resolution[1], torch.Tensor)
                    and torch.equal(resolution[1], torch.round(resolution[1]))
                )
            ), (
                "Expected each resolution side to be an int or an integer-valued "
                f"torch.Tensor. {resolution=}"
            )

        _validate_inputs()

        def _normalize_inputs(
            transform: torch.Tensor,
            resolution: Tuple[Union[int, torch.Tensor], Union[int, torch.Tensor]],
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            transform = transform.to(device=self._device, dtype=self._dtype)
            resolution = (
                torch.as_tensor(resolution[0], device=self._device, dtype=self._dtype),
                torch.as_tensor(resolution[1], device=self._device, dtype=self._dtype),
            )
            return transform, resolution

        transform, resolution = _normalize_inputs(
            transform=transform,
            resolution=resolution,
        )

        # An affine between two rasters composes only with a K stated in them, so the camera is read in pixels.
        standard = self.to(intr_convention="standard")
        # The subclass accessors, so every model hands over its focals through the one API.
        K = transform @ torch.stack(
            [
                torch.stack(
                    [standard.fx, torch.zeros_like(standard.fx), standard.cx], dim=-1
                ),
                torch.stack(
                    [torch.zeros_like(standard.fx), standard.fy, standard.cy], dim=-1
                ),
                torch.stack(
                    [
                        torch.zeros_like(standard.fx),
                        torch.zeros_like(standard.fx),
                        torch.ones_like(standard.fx),
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )
        params = self._focal_params(fx=K[..., 0, 0], fy=K[..., 1, 1])
        params["cx"], params["cy"] = K[..., 0, 2], K[..., 1, 2]
        # A single raster names the same sides for every camera of a batch.
        params["h"], params["w"] = (
            torch.broadcast_to(resolution[0], K.shape[:-2]),
            torch.broadcast_to(resolution[1], K.shape[:-2]),
        )
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

    @property
    @abstractmethod
    def fy(self) -> torch.Tensor:
        """The vertical focal length / scale, whose params key differs per model.

        Args:
            None.

        Returns:
            The vertical focal length / scale as a ``[]`` or ``[B]`` tensor.
        """

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

        def _normalize_inputs(
            device: Optional[Union[str, torch.device]],
            dtype: Optional[torch.dtype],
            intr_convention: Optional[str],
        ) -> Tuple[torch.device, torch.dtype, str]:
            # An unset device keeps this intrinsics' own.
            if device is None:
                device = self._device
            device = torch.device(device)
            # One physical device has one spelling here, so a cuda and a cuda:0 naming it never compare unequal.
            if device.type == "cuda" and device.index is None:
                # Where a tensor sent to a bare cuda lands, and so the device it reports.
                device = torch.device("cuda", torch.cuda.current_device())
            # An unset dtype keeps this intrinsics' own.
            if dtype is None:
                dtype = self._dtype
            # An unset frame keeps this intrinsics' own.
            if intr_convention is None:
                intr_convention = self._intr_convention
            return device, dtype, intr_convention

        device, dtype, intr_convention = _normalize_inputs(
            device=device,
            dtype=dtype,
            intr_convention=intr_convention,
        )

        # Nothing to restate, move, cast or copy.
        if (
            device == self._device
            and dtype == self._dtype
            and intr_convention == self._intr_convention
            and copy is False
        ):
            return self
        # The params restated on the target frame, or this intrinsics' own where the frame is unchanged; the size that change is measured against is two of those params.
        params = transform_intr_convention(
            params=self._params,
            model=type(self).MODEL,
            source_intr_convention=self._intr_convention,
            target_intr_convention=intr_convention,
        )
        # Every param moved and cast with Tensor.to's own copy semantics.
        params = apply_tensor_op(
            method="to",
            method_kwargs={
                "device": device,
                "dtype": dtype,
                "non_blocking": non_blocking,
                "copy": copy,
            },
            inputs=params,
        )
        intrinsics = type(self)(params=params, intr_convention=intr_convention)
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
        out[..., 0].div_(z).mul_(self.fx[..., None]).add_(self.cx[..., None])
        out[..., 1].div_(z).mul_(self.fx[..., None]).add_(self.cy[..., None])
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
        out[..., 0].div_(z).mul_(self.fx[..., None]).add_(self.cx[..., None])
        out[..., 1].div_(z).mul_(self.fy[..., None]).add_(self.cy[..., None])
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
        out[..., 0].mul_(self.fx[..., None]).add_(self.cx[..., None])
        out[..., 1].mul_(self.fy[..., None]).add_(self.cy[..., None])
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
        params: The model's named intrinsics parameters, every one of them sharing one leading batch shape — ``[]`` for a single camera, ``[B]`` for a batch of them; carries the resolution keys ``h`` / ``w`` alongside the projection keys.
        intr_convention: Image-plane frame the params are stated in, one of ``standard`` / ``opengl`` / ``pytorch3d`` / ``vulkan``.
        device: Optional target device for the tensor params.
        dtype: Optional target floating dtype for the tensor params.

    Returns:
        The CameraIntrinsics subclass instance for the model.
    """
    if model == "simple_pinhole":
        intrinsics = CameraIntrinsicsSimplePinhole(
            params=params,
            intr_convention=intr_convention,
            device=device,
            dtype=dtype,
        )
        return intrinsics
    if model == "pinhole":
        intrinsics = CameraIntrinsicsPinhole(
            params=params,
            intr_convention=intr_convention,
            device=device,
            dtype=dtype,
        )
        return intrinsics
    if model == "ortho":
        intrinsics = CameraIntrinsicsOrtho(
            params=params,
            intr_convention=intr_convention,
            device=device,
            dtype=dtype,
        )
        return intrinsics
    assert 0, "Should not reach here. " f"{model=}"
