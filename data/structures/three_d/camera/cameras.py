from typing import Iterator, List, Optional, Sequence, Union

import numpy as np
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
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
        device: Union[str, torch.device] = torch.device("cuda"),
    ) -> None:
        """Construct a Cameras from parallel lists of CameraIntrinsics / CameraExtrinsics.

        Args:
            intrinsics: Per-camera list of CameraIntrinsics.
            extrinsics: Per-camera list of CameraExtrinsics.
            names: Optional per-camera list of optional names.
            ids: Optional per-camera list of optional ids.
            device: Device the cameras live on, a string or torch.device.

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
        )

        self._device: torch.device = torch.device(device)
        self._intrinsics: List[CameraIntrinsics] = [
            intrinsic.to(device=self._device) for intrinsic in intrinsics
        ]
        self._extrinsics: List[CameraExtrinsics] = [
            extrinsic.to(device=self._device) for extrinsic in extrinsics
        ]
        self._names: List[Optional[str]] = names
        self._ids: List[Optional[int]] = ids
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
            return self[self._name_to_index[index]]
        if isinstance(index, slice):
            return Cameras(
                intrinsics=self._intrinsics[index],
                extrinsics=self._extrinsics[index],
                names=self._names[index],
                ids=self._ids[index],
                device=self._device,
            )
        if isinstance(index, list):
            assert index, "Index list must be non-empty"
            assert all(isinstance(item, int) for item in index), f"{index=}"
            return Cameras(
                intrinsics=[self._intrinsics[item] for item in index],
                extrinsics=[self._extrinsics[item] for item in index],
                names=[self._names[item] for item in index],
                ids=[self._ids[item] for item in index],
                device=self._device,
            )
        assert isinstance(index, int), f"{type(index)=}"
        return Camera(
            intrinsics=self._intrinsics[index],
            extrinsics=self._extrinsics[index],
            name=self._names[index],
            id=self._ids[index],
            device=self._device,
        )

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
        convention: Optional[str] = None,
    ) -> "Cameras":
        """Return this Cameras on a target device / convention.

        Args:
            device: Target device; ``None`` keeps the current device.
            convention: Target convention; ``None`` keeps each camera's convention.

        Returns:
            This Cameras when unchanged, else a new one.
        """
        # Input validations
        assert device is None or isinstance(device, (str, torch.device)), (
            "Expected target device to be None, a string, or torch.device. "
            f"{device=}"
        )
        assert convention is None or isinstance(convention, str), (
            "Expected target convention to be None or a string. " f"{convention=}"
        )

        # Input normalizations
        target_device = torch.device(device) if device is not None else self._device
        if target_device == self._device and convention is None:
            return self
        if (
            target_device == self._device
            and convention is not None
            and all(
                extrinsic.convention == convention for extrinsic in self._extrinsics
            )
        ):
            return self

        intrinsics: List[CameraIntrinsics] = []
        extrinsics: List[CameraExtrinsics] = []
        names: List[Optional[str]] = []
        ids: List[Optional[int]] = []
        for camera in self:
            moved = camera.to(device=target_device, convention=convention)
            intrinsics.append(moved.intrinsics)
            extrinsics.append(moved.extrinsics)
            names.append(moved.name)
            ids.append(moved.id)
        return Cameras(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            names=names,
            ids=ids,
            device=target_device,
        )

    def transform(
        self,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> "Cameras":
        """Return this Cameras under a similarity transform of each camera's pose.

        Args:
            scale: Similarity scale factor.
            rotation: 3x3 rotation matrix as a float32 numpy array.
            translation: Length-3 translation as a float32 numpy array.

        Returns:
            A new Cameras with each camera's CameraExtrinsics pose transformed.
        """
        intrinsics: List[CameraIntrinsics] = []
        extrinsics: List[CameraExtrinsics] = []
        names: List[Optional[str]] = []
        ids: List[Optional[int]] = []
        for camera in self:
            transformed = camera.transform(
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
    def conventions(self) -> Sequence[str]:
        """The per-camera coordinate-frame conventions, one per CameraExtrinsics.

        Args:
            None.

        Returns:
            The per-camera list of convention strings.
        """
        return [extrinsic.convention for extrinsic in self._extrinsics]

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
