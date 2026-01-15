from typing import Iterator, List, Sequence

import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.validation import (
    validate_camera_convention,
    validate_camera_extrinsics,
    validate_camera_intrinsics,
)


class Cameras:

    def __init__(
        self,
        intrinsics: torch.Tensor | List[torch.Tensor],
        extrinsics: torch.Tensor | List[torch.Tensor],
        conventions: List[str],
        names: List[str | None] | None = None,
        ids: List[int | None] | None = None,
        device: str | torch.device = torch.device("cuda"),
    ) -> None:
        # Input validations
        assert isinstance(intrinsics, (torch.Tensor, list)), f"{type(intrinsics)=}"
        assert isinstance(extrinsics, (torch.Tensor, list)), f"{type(extrinsics)=}"
        assert isinstance(conventions, list), f"{type(conventions)=}"
        assert isinstance(device, (str, torch.device)), f"{type(device)=}"

        target_device = torch.device(device)

        if isinstance(intrinsics, list):
            assert intrinsics, "intrinsics list must be non-empty"
            assert all(
                isinstance(item, torch.Tensor) for item in intrinsics
            ), f"{intrinsics=}"
            assert isinstance(
                extrinsics, list
            ), "extrinsics must be list when intrinsics is list"
            assert extrinsics, "extrinsics list must be non-empty"
            assert len(intrinsics) == len(
                extrinsics
            ), f"Batch mismatch: {len(intrinsics)=}, {len(extrinsics)=}"
            batched_intrinsics = torch.stack(
                [item.to(device=target_device) for item in intrinsics], dim=0
            )
            batched_extrinsics = torch.stack(
                [item.to(device=target_device) for item in extrinsics], dim=0
            )
        else:
            batched_intrinsics = intrinsics.to(device=target_device)
            batched_extrinsics = extrinsics.to(device=target_device)

        assert batched_intrinsics.ndim == 3, f"{batched_intrinsics.shape=}"
        assert batched_extrinsics.ndim == 3, f"{batched_extrinsics.shape=}"
        assert (
            batched_intrinsics.shape[0] == batched_extrinsics.shape[0]
        ), f"Batch mismatch: {batched_intrinsics.shape[0]=}, {batched_extrinsics.shape[0]=}"
        assert (
            len(conventions) == batched_intrinsics.shape[0]
        ), f"{len(conventions)=}, expected {batched_intrinsics.shape[0]}"

        validate_camera_intrinsics(batched_intrinsics)
        validate_camera_extrinsics(batched_extrinsics)
        for convention in conventions:
            validate_camera_convention(convention)

        self._intrinsics = batched_intrinsics
        self._extrinsics = batched_extrinsics
        self._device = target_device

        if names is None:
            names = [None] * len(conventions)
        assert len(names) == len(conventions), f"{len(names)=}, {len(conventions)=}"
        assert all(name is None or isinstance(name, str) for name in names), f"{names=}"
        self._names = list(names)

        if ids is None:
            ids = [None] * len(conventions)
        assert len(ids) == len(conventions), f"{len(ids)=}, {len(conventions)=}"
        assert all(camera_id is None or isinstance(camera_id, int) for camera_id in ids)
        self._ids = list(ids)

        self._conventions = list(conventions)

    def __len__(self) -> int:
        return self._intrinsics.shape[0]

    def __getitem__(self, index: int | slice) -> "Camera | Cameras":
        if isinstance(index, slice):
            return Cameras(
                intrinsics=self._intrinsics[index],
                extrinsics=self._extrinsics[index],
                conventions=self._conventions[index],
                names=self._names[index],
                ids=self._ids[index],
                device=self._device,
            )
        assert isinstance(index, int), f"{type(index)=}"
        return Camera(
            intrinsics=self._intrinsics[index],
            extrinsics=self._extrinsics[index],
            convention=self._conventions[index],
            name=self._names[index],
            id=self._ids[index],
            device=self._device,
        )

    def __iter__(self) -> Iterator[Camera]:
        for idx in range(len(self)):
            yield self[idx]

    @property
    def intrinsics(self) -> torch.Tensor:
        return self._intrinsics

    @property
    def extrinsics(self) -> torch.Tensor:
        return self._extrinsics

    @property
    def conventions(self) -> Sequence[str]:
        return self._conventions

    @property
    def names(self) -> Sequence[str | None]:
        return self._names

    @property
    def ids(self) -> Sequence[int | None]:
        return self._ids

    @property
    def device(self) -> torch.device:
        return self._device
