from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.extrinsics.validation import (
    validate_extr_convention,
)
from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
from data.structures.three_d.camera.intrinsics.validation import (
    validate_intr_convention,
)
from data.structures.three_d.camera.validation import validate_cameras_attributes


class Cameras:
    """An ordered collection / trajectory of cameras over a batch.

    Mirrors the two-object structure with parallel per-camera lists of
    CameraIntrinsics and CameraExtrinsics plus per-camera names / ids.
    """

    def __init__(
        self,
        intrinsics: List[CameraIntrinsics],
        extrinsics: List[CameraExtrinsics],
        names: Optional[List[Optional[str]]] = None,
        ids: Optional[List[Optional[int]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Construct a Cameras from parallel lists of CameraIntrinsics / CameraExtrinsics.

        Args:
            intrinsics: Per-camera list of CameraIntrinsics.
            extrinsics: Per-camera list of CameraExtrinsics.
            names: Optional per-camera list of optional names.
            ids: Optional per-camera list of optional ids.
            device: Optional target device for the cameras' tensors.
            dtype: Optional target floating dtype for the cameras' tensors.

        Returns:
            None.
        """
        # Input normalizations
        names = names if names is not None else [None] * len(intrinsics)
        ids = ids if ids is not None else [None] * len(intrinsics)

        validate_cameras_attributes(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            names=names,
            ids=ids,
            device=device,
            dtype=dtype,
        )

        if device is not None or dtype is not None:
            intrinsics = [
                intrinsic.to(device=device, dtype=dtype) for intrinsic in intrinsics
            ]
            extrinsics = [
                extrinsic.to(device=device, dtype=dtype) for extrinsic in extrinsics
            ]
        component_device = (
            intrinsics[0].device
            if intrinsics
            else torch.device(device) if device is not None else torch.device("cpu")
        )
        component_dtype = (
            intrinsics[0].dtype if intrinsics else dtype or torch.get_default_dtype()
        )
        for intrinsic, extrinsic in zip(intrinsics, extrinsics):
            assert intrinsic.device == component_device, (
                "Expected all CameraIntrinsics entries to share device. "
                f"{intrinsic.device=} {component_device=}"
            )
            assert extrinsic.device == component_device, (
                "Expected all CameraExtrinsics entries to share device. "
                f"{extrinsic.device=} {component_device=}"
            )
            assert intrinsic.dtype == component_dtype, (
                "Expected all CameraIntrinsics entries to share dtype. "
                f"{intrinsic.dtype=} {component_dtype=}"
            )
            assert extrinsic.dtype == component_dtype, (
                "Expected all CameraExtrinsics entries to share dtype. "
                f"{extrinsic.dtype=} {component_dtype=}"
            )
        self._intrinsics: List[CameraIntrinsics] = intrinsics
        self._extrinsics: List[CameraExtrinsics] = extrinsics
        self._names: List[Optional[str]] = names
        self._ids: List[Optional[int]] = ids
        self._device: torch.device = component_device
        self._dtype: torch.dtype = component_dtype
        self._name_to_index = {name: index for index, name in enumerate(self._names)}

    def __len__(self) -> int:
        """The number of cameras in the collection.

        Args:
            None.

        Returns:
            The number of cameras.
        """
        return len(self._intrinsics)

    def __getitem__(
        self, index: Union[int, slice, List[int], str]
    ) -> Union["Camera", "Cameras"]:
        """Index the collection.

        A name / int yields one Camera, a slice / int-list yields a sub-Cameras.

        Args:
            index: A name string, an int, a slice, or a list of ints.

        Returns:
            A single Camera or a sub-Cameras collection.
        """
        if isinstance(index, str):
            assert index in self._name_to_index, f"Camera name '{index}' not found"
            camera_index = self._name_to_index[index]
            return Camera(
                intrinsics=self._intrinsics[camera_index],
                extrinsics=self._extrinsics[camera_index],
                name=self._names[camera_index],
                id=self._ids[camera_index],
                device=self._device,
                dtype=self._dtype,
            )
        if isinstance(index, (slice, list)):
            if isinstance(index, slice):
                intrinsics = self._intrinsics[index]
                extrinsics = self._extrinsics[index]
                names = self._names[index]
                ids = self._ids[index]
            else:
                assert index, "Index list must be non-empty"
                assert all(isinstance(item, int) for item in index), f"{index=}"
                intrinsics = [self._intrinsics[item] for item in index]
                extrinsics = [self._extrinsics[item] for item in index]
                names = [self._names[item] for item in index]
                ids = [self._ids[item] for item in index]
            return Cameras(
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                names=names,
                ids=ids,
                device=self._device,
                dtype=self._dtype,
            )
        if isinstance(index, int):
            return Camera(
                intrinsics=self._intrinsics[index],
                extrinsics=self._extrinsics[index],
                name=self._names[index],
                id=self._ids[index],
                device=self._device,
                dtype=self._dtype,
            )
        assert 0, "Should not reach here. " f"{type(index)=} {index=}"

    def __iter__(self) -> Iterator["Camera"]:
        """Iterate the collection one Camera at a time.

        Args:
            None.

        Returns:
            An iterator over the per-index Camera objects.
        """
        for index in range(len(self)):
            yield self[index]

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
        intr_convention: Optional[str] = None,
        extr_convention: Optional[str] = None,
    ) -> "Cameras":
        """Return this Cameras with tensor placement and convention changes.

        Each half is named exactly as the Camera it batches names it.

        Args:
            device: Target device; ``None`` keeps the current device.
            dtype: Target floating dtype; ``None`` keeps the current dtype.
            non_blocking: Whether tensor moves may be asynchronous.
            copy: Whether tensor moves must allocate new storage even when unchanged.
            intr_convention: Target image-plane frame; ``None`` keeps each camera's own.
            extr_convention: Target pose frame; ``None`` keeps each camera's own.

        Returns:
            This Cameras when unchanged, else a new one.
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
            assert intr_convention is None or isinstance(intr_convention, str), (
                "Expected target image-plane frame to be None or a string. "
                f"{intr_convention=}"
            )
            if intr_convention is not None:
                validate_intr_convention(intr_convention=intr_convention)
            assert extr_convention is None or isinstance(extr_convention, str), (
                "Expected target pose frame to be None or a string. "
                f"{extr_convention=}"
            )
            if extr_convention is not None:
                validate_extr_convention(extr_convention=extr_convention)

        _validate_inputs()

        def _normalize_inputs(
            device: Optional[Union[str, torch.device]],
            dtype: Optional[torch.dtype],
        ) -> Tuple[torch.device, torch.dtype]:
            device = torch.device(device) if device is not None else self._device
            dtype = dtype if dtype is not None else self._dtype
            return device, dtype

        device, dtype = _normalize_inputs(
            device=device,
            dtype=dtype,
        )

        if (
            device == self._device
            and dtype == self._dtype
            and (
                intr_convention is None
                or all(
                    intrinsic.intr_convention == intr_convention
                    for intrinsic in self._intrinsics
                )
            )
            and (
                extr_convention is None
                or all(
                    extrinsic.extr_convention == extr_convention
                    for extrinsic in self._extrinsics
                )
            )
            and copy is False
        ):
            return self

        intrinsics: List[CameraIntrinsics] = []
        extrinsics: List[CameraExtrinsics] = []
        names: List[Optional[str]] = []
        ids: List[Optional[int]] = []
        for camera in self:
            moved = camera.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
                copy=copy,
                intr_convention=intr_convention,
                extr_convention=extr_convention,
            )
            intrinsics.append(moved.intrinsics)
            extrinsics.append(moved.extrinsics)
            names.append(moved.name)
            ids.append(moved.id)
        return Cameras(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            names=names,
            ids=ids,
            device=device,
            dtype=dtype,
        )

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
        """Return this Cameras under a similarity transform of each camera's pose.

        Args:
            scale: Scalar similarity scale factor as a number, numpy array, or torch.Tensor.
            rotation: 3x3 rotation matrix as a numpy array, torch.Tensor, or nested numeric list.
            translation: Length-3 translation as a numpy array, torch.Tensor, tuple, or list.

        Returns:
            A new Cameras with each camera's CameraExtrinsics pose transformed.
        """
        intrinsics: List[CameraIntrinsics] = []
        extrinsics: List[CameraExtrinsics] = []
        names: List[Optional[str]] = []
        ids: List[Optional[int]] = []
        for camera in self:
            transformed = camera.transform_extrinsics(
                scale=scale,
                rotation=rotation,
                translation=translation,
            )
            intrinsics.append(transformed.intrinsics)
            extrinsics.append(transformed.extrinsics)
            names.append(transformed.name)
            ids.append(transformed.id)
        return Cameras(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            names=names,
            ids=ids,
            device=self._device,
            dtype=self._dtype,
        )

    @property
    def intrinsics(self) -> Sequence[CameraIntrinsics]:
        """The per-camera CameraIntrinsics.

        Args:
            None.

        Returns:
            The per-camera list of CameraIntrinsics.
        """
        return self._intrinsics

    @property
    def extrinsics(self) -> Sequence[CameraExtrinsics]:
        """The per-camera CameraExtrinsics.

        Args:
            None.

        Returns:
            The per-camera list of CameraExtrinsics.
        """
        return self._extrinsics

    @property
    def names(self) -> Sequence[Optional[str]]:
        """The per-camera names.

        Args:
            None.

        Returns:
            The per-camera list of optional names.
        """
        return self._names

    @property
    def ids(self) -> Sequence[Optional[int]]:
        """The per-camera ids.

        Args:
            None.

        Returns:
            The per-camera list of optional ids.
        """
        return self._ids

    @property
    def device(self) -> torch.device:
        """The device the cameras live on.

        Args:
            None.

        Returns:
            The device.
        """
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """The dtype shared by the cameras' tensor state.

        Args:
            None.

        Returns:
            The torch dtype shared by every camera component.
        """
        return self._dtype

    @property
    def center(self) -> torch.Tensor:
        """The [N, 3] stack of per-camera centers.

        Args:
            None.

        Returns:
            The ``[N, 3]`` per-camera centers torch.Tensor.
        """
        centers = torch.stack(
            [extrinsic.center for extrinsic in self._extrinsics], dim=0
        )
        assert centers.shape == (len(self), 3), f"{centers.shape=}"
        return centers

    @property
    def right(self) -> torch.Tensor:
        """The [N, 3] stack of per-camera physical right axes.

        Args:
            None.

        Returns:
            The ``[N, 3]`` per-camera right axes torch.Tensor.
        """
        vecs = torch.stack([extrinsic.right for extrinsic in self._extrinsics], dim=0)
        assert vecs.shape == (len(self), 3), f"{vecs.shape=}"
        return vecs

    @property
    def forward(self) -> torch.Tensor:
        """The [N, 3] stack of per-camera physical forward axes.

        Args:
            None.

        Returns:
            The ``[N, 3]`` per-camera forward axes torch.Tensor.
        """
        vecs = torch.stack([extrinsic.forward for extrinsic in self._extrinsics], dim=0)
        assert vecs.shape == (len(self), 3), f"{vecs.shape=}"
        return vecs

    @property
    def up(self) -> torch.Tensor:
        """The [N, 3] stack of per-camera physical up axes.

        Args:
            None.

        Returns:
            The ``[N, 3]`` per-camera up axes torch.Tensor.
        """
        vecs = torch.stack([extrinsic.up for extrinsic in self._extrinsics], dim=0)
        assert vecs.shape == (len(self), 3), f"{vecs.shape=}"
        return vecs
