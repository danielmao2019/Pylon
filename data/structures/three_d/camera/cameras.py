from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    CameraIntrinsics,
)
from data.structures.three_d.camera.validation import validate_cameras_attributes


class Cameras:
    """A batch of cameras: one CameraIntrinsics and one CameraExtrinsics carrying a leading batch axis.

    Every method the two components already have operates on the whole batch, so the batch never loops over its own cameras to compute anything.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        names: Optional[List[Optional[str]]] = None,
        ids: Optional[List[Optional[int]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Construct a Cameras from a batched CameraIntrinsics and a batched CameraExtrinsics.

        Args:
            intrinsics: Batched CameraIntrinsics whose params are each a ``[B]`` torch.Tensor.
            extrinsics: Batched CameraExtrinsics whose camera-to-world matrix is a ``[B, 4, 4]`` torch.Tensor.
            names: Optional per-camera list of optional names, parallel to the batch axis.
            ids: Optional per-camera list of optional ids, parallel to the batch axis.
            device: Optional target device for the batch's tensors; ``None`` resolves to the given extrinsics' own device.
            dtype: Optional target floating dtype for the batch's tensors; ``None`` resolves to the given extrinsics' own dtype.

        Returns:
            None.
        """

        def _validate_inputs() -> None:
            validate_cameras_attributes(
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                names=names,
                ids=ids,
                device=device,
                dtype=dtype,
            )

        _validate_inputs()

        def _normalize_inputs(
            intrinsics: CameraIntrinsics,
            extrinsics: CameraExtrinsics,
            names: Optional[List[Optional[str]]],
            ids: Optional[List[Optional[int]]],
            device: Optional[Union[str, torch.device]],
            dtype: Optional[torch.dtype],
        ) -> Tuple[
            CameraIntrinsics,
            CameraExtrinsics,
            List[Optional[str]],
            List[Optional[int]],
            torch.device,
            torch.dtype,
        ]:
            if device is None:
                # The one exception: an unset device resolves to the given extrinsics'.
                device = extrinsics.device
            # One physical device has one spelling here, so a cuda and a cuda:0 naming it never compare unequal.
            device = torch.device(device)
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            if dtype is None:
                # The one exception: an unset dtype resolves to the given extrinsics'.
                dtype = extrinsics.dtype
            # Both components are brought to the resolved device and dtype, never the other way around.
            intrinsics = intrinsics.to(device=device, dtype=dtype)
            extrinsics = extrinsics.to(device=device, dtype=dtype)
            batch_size = len(extrinsics.extrinsics)
            names = names if names is not None else [None] * batch_size
            ids = ids if ids is not None else [None] * batch_size
            return intrinsics, extrinsics, names, ids, device, dtype

        intrinsics, extrinsics, names, ids, device, dtype = _normalize_inputs(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            names=names,
            ids=ids,
            device=device,
            dtype=dtype,
        )

        self._intrinsics: CameraIntrinsics = intrinsics
        self._extrinsics: CameraExtrinsics = extrinsics
        self._names: List[Optional[str]] = names
        self._ids: List[Optional[int]] = ids
        self._name_to_index = {}
        for index, name in enumerate(names):
            if name is None:
                continue
            assert name not in self._name_to_index, (
                "Expected every Cameras name to be unique, since a name two cameras "
                "share cannot resolve to one of them. "
                f"{name=} {self._name_to_index[name]=} {index=} {names=}"
            )
            self._name_to_index[name] = index
        self._device: torch.device = device
        self._dtype: torch.dtype = dtype

    @property
    def intrinsics(self) -> CameraIntrinsics:
        """The batch's CameraIntrinsics, whose params carry the batch axis.

        Args:
            None.

        Returns:
            The batched CameraIntrinsics, whose own project / scale_intrinsics cover every camera at once.
        """
        return self._intrinsics

    @property
    def extrinsics(self) -> CameraExtrinsics:
        """The batch's CameraExtrinsics, whose matrix carries the batch axis.

        Args:
            None.

        Returns:
            The batched CameraExtrinsics, whose ``[B, 4, 4]`` matrix every pose op broadcasts over.
        """
        return self._extrinsics

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
        intr_convention: Optional[str] = None,
        extr_convention: Optional[str] = None,
    ) -> "Cameras":
        """Return this batch with tensor placement / copy semantics plus optional frame conversions.

        Each half is delegated to the component that owns it.

        Args:
            device: Target device; ``None`` keeps the current device.
            dtype: Target floating dtype; ``None`` keeps the current dtype.
            non_blocking: Whether tensor moves may be asynchronous.
            copy: Whether tensor moves must allocate new storage even when unchanged.
            intr_convention: Target image-plane frame; ``None`` keeps the batch's own.
            extr_convention: Target pose frame; ``None`` keeps the batch's own.

        Returns:
            A Cameras whose components carry the requested placement and frames.
        """
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
        # The requested placement is handed on, so the new batch's device and dtype follow it.
        cameras = Cameras(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            names=self._names,
            ids=self._ids,
            device=device,
            dtype=dtype,
        )
        return cameras

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
    ) -> "Cameras":
        """Return this batch restated against a different resolution.

        The rescale is elementwise on the ``[B]`` params the intrinsics already holds.

        Args:
            resolution: Target raster as an int, a ``(h, w)`` tuple / list, or an array-like of two values.
            scale: Multiplicative factor as a number, a two-value tuple / list, or an array-like of two values.

        Returns:
            A new Cameras whose CameraIntrinsics is stated against the requested raster.
        """
        intrinsics = self._intrinsics.scale_intrinsics(
            resolution=resolution,
            scale=scale,
        )
        cameras = Cameras(
            intrinsics=intrinsics,
            extrinsics=self._extrinsics,
            names=self._names,
            ids=self._ids,
        )
        return cameras

    def transform_intrinsics(
        self,
        transform: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> "Cameras":
        """Return this batch with its intrinsics restated by a pixel-frame affine.

        The affine broadcasts over the batch axis.

        Args:
            transform: ``[3, 3]`` pixel-frame affine torch.Tensor mapping this image's pixels onto the other image's.
            resolution: The other image's ``(h, w)`` raster.

        Returns:
            A new Cameras whose CameraIntrinsics is restated onto that other image.
        """
        intrinsics = self._intrinsics.transform_intrinsics(
            transform=transform,
            resolution=resolution,
        )
        cameras = Cameras(
            intrinsics=intrinsics,
            extrinsics=self._extrinsics,
            names=self._names,
            ids=self._ids,
        )
        return cameras

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
    ) -> "Cameras":
        """Return this batch under a similarity transform applied to every pose at once.

        Args:
            scale: Scalar similarity scale factor as a number, numpy array, or torch.Tensor.
            rotation: 3x3 rotation matrix as a numpy array, torch.Tensor, or nested numeric list.
            translation: Length-3 translation as a numpy array, torch.Tensor, tuple, or list.

        Returns:
            A new Cameras whose CameraExtrinsics poses are all transformed.
        """
        extrinsics = self._extrinsics.transform_extrinsics(
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        cameras = Cameras(
            intrinsics=self._intrinsics,
            extrinsics=extrinsics,
            names=self._names,
            ids=self._ids,
        )
        return cameras

    def __len__(self) -> int:
        """The number of cameras in the batch.

        Args:
            None.

        Returns:
            The extent of the components' leading batch axis.
        """
        return self._extrinsics.extrinsics.shape[0]

    def __getitem__(
        self, index: Union[int, slice, List[int], str]
    ) -> Union["Camera", "Cameras"]:
        """Index the batch by slicing the leading axis of both components.

        Never by selecting from stored per-camera objects: a name / int yields one Camera, a slice / int-list yields a sub-Cameras.

        Args:
            index: A name string, an int, a slice, or a list of ints.

        Returns:
            A single Camera or a sub-Cameras batch.
        """
        if isinstance(index, str):
            assert index in self._name_to_index, (
                "Expected the indexing name to label a camera in this batch. "
                f"{index=} {list(self._name_to_index.keys())=}"
            )
            index = self._name_to_index[index]
        intrinsics = self._intrinsics[index]
        extrinsics = self._extrinsics[index]
        if isinstance(index, int):
            return Camera(
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                name=self._names[index],
                id=self._ids[index],
            )
        names = (
            self._names[index]
            if isinstance(index, slice)
            else [self._names[item] for item in index]
        )
        ids = (
            self._ids[index]
            if isinstance(index, slice)
            else [self._ids[item] for item in index]
        )
        return Cameras(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            names=names,
            ids=ids,
        )

    def __iter__(self) -> Iterator["Camera"]:
        """Iterate one Camera at a time.

        For callers that genuinely need a single camera rather than the batch.

        Args:
            None.

        Returns:
            An iterator over the per-index Camera objects.
        """
        for index in range(len(self)):
            yield self[index]

    @property
    def names(self) -> Sequence[Optional[str]]:
        """The per-camera labels.

        Args:
            None.

        Returns:
            The per-camera list of optional names, metadata that never enters a tensor op.
        """
        return self._names

    @property
    def ids(self) -> Sequence[Optional[int]]:
        """The per-camera integer identities.

        Args:
            None.

        Returns:
            The per-camera list of optional ids, which survive a serialize / deserialize round trip.
        """
        return self._ids

    @property
    def device(self) -> torch.device:
        """The device this batch was constructed on.

        Args:
            None.

        Returns:
            The torch.device its component tensors were brought to at construction.
        """
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """The dtype this batch was constructed with.

        Args:
            None.

        Returns:
            The floating torch dtype its component tensors were cast to at construction.
        """
        return self._dtype

    @property
    def center(self) -> torch.Tensor:
        """The batch's camera centers.

        Args:
            None.

        Returns:
            The ``[B, 3]`` camera centers torch.Tensor, its extrinsics' own center under the batch axis.
        """
        return self._extrinsics.center

    @property
    def right(self) -> torch.Tensor:
        """The batch's physical right axes.

        Args:
            None.

        Returns:
            The ``[B, 3]`` unit right axes torch.Tensor, its extrinsics' own right under the batch axis.
        """
        return self._extrinsics.right

    @property
    def forward(self) -> torch.Tensor:
        """The batch's physical forward axes.

        Args:
            None.

        Returns:
            The ``[B, 3]`` unit forward axes torch.Tensor, its extrinsics' own forward under the batch axis.
        """
        return self._extrinsics.forward

    @property
    def up(self) -> torch.Tensor:
        """The batch's physical up axes.

        Args:
            None.

        Returns:
            The ``[B, 3]`` unit up axes torch.Tensor, its extrinsics' own up under the batch axis.
        """
        return self._extrinsics.up
