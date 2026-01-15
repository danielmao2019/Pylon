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
        intrinsics: torch.Tensor | List[torch.Tensor | None] | None,
        extrinsics: torch.Tensor | List[torch.Tensor],
        conventions: List[str],
        names: List[str | None] | None = None,
        ids: List[int | None] | None = None,
        device: str | torch.device = torch.device("cuda"),
    ) -> None:
        # Input validations
        assert (
            intrinsics is None
            or isinstance(intrinsics, torch.Tensor)
            or (
                isinstance(intrinsics, list)
                and all(
                    _intr is None or isinstance(_intr, torch.Tensor)
                    for _intr in intrinsics
                )
            )
        ), f"{type(intrinsics)=}"
        assert isinstance(extrinsics, torch.Tensor) or (
            isinstance(extrinsics, list)
            and all(isinstance(_extr, torch.Tensor) for _extr in extrinsics)
        ), f"{type(extrinsics)=}"
        assert isinstance(conventions, list), f"{type(conventions)=}"
        for convention in conventions:
            validate_camera_convention(convention)
        assert isinstance(device, (str, torch.device)), f"{type(device)=}"
        assert names is None or (
            isinstance(names, list)
            and all(_n is None or isinstance(_n, str) for _n in names)
        ), f""
        assert ids is None or (
            isinstance(ids, list)
            and all(_i is None or isinstance(_i, int) for _i in ids)
        ), f""

        # Input normalization

        batch_size = len(extrinsics)
        if isinstance(intrinsics, torch.Tensor):
            batched_intrinsics = intrinsics.to(device=device)
        elif intrinsics is None:
            batched_intrinsics = (
                torch.eye(3, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )
        elif isinstance(intrinsics, list):
            assert intrinsics, "intrinsics list must be non-empty"
            intr_stack = []
            for item in intrinsics:
                assert item is None or isinstance(item, torch.Tensor), f"{type(item)=}"
                if item is None:
                    intr_stack.append(torch.eye(3, dtype=torch.float32, device=device))
                else:
                    intr_stack.append(item.to(device=device))
            batched_intrinsics = torch.stack(intr_stack, dim=0)
        else:
            assert False, f"Unsupported intrinsics type: {type(intrinsics)}"
        assert batched_intrinsics.ndim == 3, f"{batched_intrinsics.shape=}"
        validate_camera_intrinsics(batched_intrinsics)

        if isinstance(extrinsics, torch.Tensor):
            batched_extrinsics = extrinsics.to(device=device)
        elif isinstance(extrinsics, list):
            assert extrinsics, "extrinsics list must be non-empty"
            batched_extrinsics = torch.stack(
                [item.to(device=device) for item in extrinsics], dim=0
            )
        else:
            assert 0, f""
        assert batched_extrinsics.ndim == 3, f""
        validate_camera_extrinsics(batched_extrinsics)

        if names is None:
            names = [None] * len(conventions)
        assert isinstance(names, list)

        if ids is None:
            ids = [None] * len(conventions)
        assert isinstance(ids, list)

        target_device = torch.device(device)

        # Compatibility checks
        assert (
            batched_intrinsics.shape[0] == batch_size
        ), f"Batch mismatch: {batched_intrinsics.shape[0]=}, {batch_size=}"
        assert (
            len(conventions) == batch_size
        ), f"{len(conventions)=}, expected {batch_size}"
        assert len(names) == batch_size, f"{len(names)=}, {batch_size=}"
        assert len(ids) == batch_size, f"{len(ids)=}, {batch_size=}"

        self._intrinsics = batched_intrinsics
        self._extrinsics = batched_extrinsics
        self._conventions = conventions
        self._names = names
        self._ids = ids
        self._device = target_device

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

    @property
    def right(self) -> torch.Tensor:
        vecs = self._extrinsics[:, :3, 0]
        norms = torch.linalg.norm(vecs, dim=1)
        assert torch.allclose(
            norms,
            torch.ones_like(norms),
            atol=1.0e-05,
            rtol=0.0,
        ), f"Right vectors must be unit, norms {norms}"
        return vecs

    @property
    def forward(self) -> torch.Tensor:
        vecs = torch.empty_like(self._extrinsics[:, :3, 0])
        for idx, convention in enumerate(self._conventions):
            if convention == "standard":
                vecs[idx] = self._extrinsics[idx, :3, 1]
            elif convention == "opengl":
                vecs[idx] = -self._extrinsics[idx, :3, 2]
            elif convention == "opencv":
                vecs[idx] = self._extrinsics[idx, :3, 2]
            elif convention == "pytorch3d":
                vecs[idx] = self._extrinsics[idx, :3, 1]
            else:
                assert False, f"Unsupported convention: {convention}"
        norms = torch.linalg.norm(vecs, dim=1)
        assert torch.allclose(
            norms,
            torch.ones_like(norms),
            atol=1.0e-05,
            rtol=0.0,
        ), f"Forward vectors must be unit, norms {norms}"
        return vecs

    @property
    def up(self) -> torch.Tensor:
        vecs = torch.empty_like(self._extrinsics[:, :3, 0])
        for idx, convention in enumerate(self._conventions):
            if convention == "standard":
                vecs[idx] = self._extrinsics[idx, :3, 2]
            elif convention == "opengl":
                vecs[idx] = self._extrinsics[idx, :3, 1]
            elif convention == "opencv":
                vecs[idx] = -self._extrinsics[idx, :3, 1]
            elif convention == "pytorch3d":
                vecs[idx] = self._extrinsics[idx, :3, 2]
            else:
                assert False, f"Unsupported convention: {convention}"
        norms = torch.linalg.norm(vecs, dim=1)
        assert torch.allclose(
            norms,
            torch.ones_like(norms),
            atol=1.0e-05,
            rtol=0.0,
        ), f"Up vectors must be unit, norms {norms}"
        return vecs
