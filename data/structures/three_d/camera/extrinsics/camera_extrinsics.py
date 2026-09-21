from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from data.structures.three_d.camera.extrinsics.conventions import (
    transform_extr_convention,
)
from data.structures.three_d.camera.extrinsics.validation import (
    validate_camera_extrinsics_attributes,
    validate_extr_convention,
    validate_rotation_matrix,
    validate_translation_vector,
)

_ORTHOGONALITY_REPAIR_ATOL = 1.0e-05


class CameraExtrinsics:
    """A camera's pose: a 4x4 cam2world matrix plus the pose frame it is expressed in."""

    def __init__(
        self,
        extrinsics: Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]],
        extr_convention: str,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Construct a CameraExtrinsics from a 4x4 cam2world matrix and its pose frame.

        Args:
            extrinsics: Camera-to-world extrinsics matrix as a numpy array, torch.Tensor, or nested numeric list, a ``[4, 4]`` for one camera or a ``[B, 4, 4]`` for a batch of them.
            extr_convention: Coordinate-frame convention string.
            device: Optional target device for the extrinsics tensor; ``None`` resolves to the given matrix's own device, cpu for a numpy array or nested list.
            dtype: Optional target floating dtype for the extrinsics tensor; ``None`` resolves to the given matrix's own dtype, float32 for a nested list.

        Returns:
            None.
        """
        validate_camera_extrinsics_attributes(
            extrinsics=extrinsics,
            extr_convention=extr_convention,
            device=device,
            dtype=dtype,
        )

        def _normalize_inputs(
            extrinsics: Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]],
            device: Optional[Union[str, torch.device]],
            dtype: Optional[torch.dtype],
        ) -> Tuple[torch.Tensor, torch.device, torch.dtype]:
            if device is None:
                if isinstance(extrinsics, torch.Tensor):
                    # The one exception: an unset device resolves to the given matrix's, so a component __getitem__ rebuilds stays where its batch is.
                    device = extrinsics.device
                else:
                    device = torch.device("cpu")
            device = torch.device(device)
            # One physical device has one spelling here, so a cuda and a cuda:0 naming it never compare unequal.
            if device.type == "cuda" and device.index is None:
                # Where a tensor sent to a bare cuda lands, and so the device it reports.
                device = torch.device("cuda", torch.cuda.current_device())
            if dtype is None:
                if isinstance(extrinsics, (torch.Tensor, np.ndarray)):
                    # The one exception: an unset dtype resolves to the given matrix's, so a component __getitem__ rebuilds keeps the dtype its batch holds.
                    dtype = torch.as_tensor(extrinsics).dtype
                else:
                    dtype = torch.float32
            # The matrix follows the resolved device and dtype, never the other way around.
            extrinsics = torch.as_tensor(extrinsics, device=device, dtype=dtype)
            return extrinsics, device, dtype

        extrinsics, device, dtype = _normalize_inputs(
            extrinsics=extrinsics,
            device=device,
            dtype=dtype,
        )

        # None where the matrix is a [4, 4]: an unbatched extrinsics carries no batch axis, and states one camera.
        batch_size = extrinsics.shape[0] if extrinsics.ndim == 3 else None
        self._extrinsics: torch.Tensor = extrinsics
        self._extr_convention: str = extr_convention
        self._device: torch.device = device
        self._dtype: torch.dtype = dtype
        self._batch_size: Optional[int] = batch_size

    @property
    def extrinsics(self) -> torch.Tensor:
        """The 4x4 camera-to-world extrinsics matrix.

        Args:
            None.

        Returns:
            The 4x4 camera-to-world extrinsics torch.Tensor.
        """
        return self._extrinsics

    @property
    def extr_convention(self) -> str:
        """The pose frame this cam2world matrix is expressed in.

        Args:
            None.

        Returns:
            The pose-frame convention string (standard / opengl / opencv / pytorch3d / arkit).
        """
        return self._extr_convention

    @property
    def device(self) -> torch.device:
        """The device the extrinsics live on.

        Args:
            None.

        Returns:
            The device.
        """
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the extrinsics tensor.

        Args:
            None.

        Returns:
            The torch dtype of the camera-to-world matrix.
        """
        return self._dtype

    @property
    def is_batched(self) -> bool:
        """Whether the cam2world matrix carries a batch axis.

        Args:
            None.

        Returns:
            True for a ``[B, 4, 4]`` matrix, False for the ``[4, 4]`` of the one camera an unbatched extrinsics states.
        """
        return self._batch_size is not None

    def __len__(self) -> int:
        """The extent of the batch axis this extrinsics carries.

        Args:
            None.

        Returns:
            The number ``B`` of cameras a batched ``[B, 4, 4]`` matrix carries; an unbatched extrinsics has no length.
        """
        # A [4, 4] matrix carries no batch axis, so it has no length.
        assert self._batch_size is not None, (
            "Expected a batched CameraExtrinsics, since an unbatched [4, 4] matrix "
            f"carries no batch axis and so has no length. {self._extrinsics.shape=}"
        )
        return self._batch_size

    def __getitem__(
        self, index: Union[int, slice, List[int], None]
    ) -> "CameraExtrinsics":
        """Index the leading batch axis the camera-to-world matrix carries.

        Args:
            index: The index applied to the matrix's leading axis the way the tensor indexes its own, so ``None`` adds an axis of one and an int drops it.

        Returns:
            A CameraExtrinsics whose camera-to-world matrix carries the indexed leading axis.
        """
        extrinsics = CameraExtrinsics(
            extrinsics=self._extrinsics[index],
            extr_convention=self._extr_convention,
        )
        return extrinsics

    @property
    def w2c(self) -> torch.Tensor:
        """The world-to-camera matrix (inverse of extrinsics).

        Args:
            None.

        Returns:
            The 4x4 world-to-camera torch.Tensor.
        """
        w2c = torch.inverse(self._extrinsics)
        return w2c

    @property
    def center(self) -> torch.Tensor:
        """The camera center.

        Args:
            None.

        Returns:
            The camera center ``extrinsics[..., :3, 3]`` as a ``[..., 3]`` torch.Tensor, one per camera the matrix carries.
        """
        return self._extrinsics[..., :3, 3]

    @property
    def right(self) -> torch.Tensor:
        """The extr_convention-dispatched physical right axis.

        Args:
            None.

        Returns:
            The unit right axis as a ``[..., 3]`` torch.Tensor, one per camera the matrix carries.
        """
        if self._extr_convention == "standard":
            vec = self._extrinsics[..., :3, 0]
        elif self._extr_convention == "opengl":
            vec = self._extrinsics[..., :3, 0]
        elif self._extr_convention == "opencv":
            vec = self._extrinsics[..., :3, 0]
        elif self._extr_convention == "pytorch3d":
            vec = -self._extrinsics[..., :3, 0]
        elif self._extr_convention == "arkit":
            vec = -self._extrinsics[..., :3, 1]
        else:
            assert False, f"Unsupported extr_convention: {self._extr_convention}"
        norm = torch.linalg.norm(vec, dim=-1)
        assert torch.allclose(
            input=norm,
            other=torch.ones_like(norm),
            rtol=0.0,
            atol=1.0e-05,
        ), f"Right vector must be unit, got norm {norm}"
        return vec

    @property
    def forward(self) -> torch.Tensor:
        """The extr_convention-dispatched physical forward axis.

        Args:
            None.

        Returns:
            The unit forward axis as a ``[..., 3]`` torch.Tensor, one per camera the matrix carries.
        """
        if self._extr_convention == "standard":
            vec = self._extrinsics[..., :3, 1]
        elif self._extr_convention == "opengl":
            vec = -self._extrinsics[..., :3, 2]
        elif self._extr_convention == "opencv":
            vec = self._extrinsics[..., :3, 2]
        elif self._extr_convention == "pytorch3d":
            vec = self._extrinsics[..., :3, 2]
        elif self._extr_convention == "arkit":
            vec = self._extrinsics[..., :3, 2]
        else:
            assert False, f"Unsupported extr_convention: {self._extr_convention}"
        norm = torch.linalg.norm(vec, dim=-1)
        assert torch.allclose(
            input=norm,
            other=torch.ones_like(norm),
            rtol=0.0,
            atol=1.0e-05,
        ), f"Forward vector must be unit, got norm {norm}"
        return vec

    @property
    def up(self) -> torch.Tensor:
        """The extr_convention-dispatched physical up axis.

        Args:
            None.

        Returns:
            The unit up axis as a ``[..., 3]`` torch.Tensor, one per camera the matrix carries.
        """
        if self._extr_convention == "standard":
            vec = self._extrinsics[..., :3, 2]
        elif self._extr_convention == "opengl":
            vec = self._extrinsics[..., :3, 1]
        elif self._extr_convention == "opencv":
            vec = -self._extrinsics[..., :3, 1]
        elif self._extr_convention == "pytorch3d":
            vec = self._extrinsics[..., :3, 1]
        elif self._extr_convention == "arkit":
            vec = -self._extrinsics[..., :3, 0]
        else:
            assert False, f"Unsupported extr_convention: {self._extr_convention}"
        norm = torch.linalg.norm(vec, dim=-1)
        assert torch.allclose(
            input=norm,
            other=torch.ones_like(norm),
            rtol=0.0,
            atol=1.0e-05,
        ), f"Up vector must be unit, got norm {norm}"
        return vec

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
        extr_convention: Optional[str] = None,
    ) -> "CameraExtrinsics":
        """Return this CameraExtrinsics with tensor placement and pose-frame changes.

        Args:
            device: Target device; ``None`` keeps the current device.
            dtype: Target floating dtype; ``None`` keeps the current dtype.
            non_blocking: Whether tensor moves may be asynchronous.
            copy: Whether tensor moves must allocate new storage even when unchanged.
            extr_convention: Target pose frame; ``None`` keeps the current one.

        Returns:
            This CameraExtrinsics when unchanged, else a new one.
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
            assert extr_convention is None or isinstance(extr_convention, str), (
                "Expected target pose frame to be None or a string. "
                f"{extr_convention=}"
            )
            if extr_convention is not None:
                validate_extr_convention(extr_convention)

        _validate_inputs()

        def _normalize_inputs(
            device: Optional[Union[str, torch.device]],
            dtype: Optional[torch.dtype],
            extr_convention: Optional[str],
        ) -> Tuple[torch.device, torch.dtype, str]:
            device = torch.device(device) if device is not None else self._device
            dtype = dtype if dtype is not None else self._dtype
            extr_convention = (
                extr_convention
                if extr_convention is not None
                else self._extr_convention
            )
            return device, dtype, extr_convention

        device, dtype, extr_convention = _normalize_inputs(
            device=device,
            dtype=dtype,
            extr_convention=extr_convention,
        )

        if (
            device == self._device
            and dtype == self._dtype
            and extr_convention == self._extr_convention
            and copy is False
        ):
            return self

        if extr_convention != self._extr_convention:
            extrinsics = transform_extr_convention(
                camera_extrinsics=self,
                target_extr_convention=extr_convention,
            )
        else:
            extrinsics = self._extrinsics

        extrinsics = extrinsics.to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
            copy=copy,
        )
        extrinsics = CameraExtrinsics(
            extrinsics=extrinsics,
            extr_convention=extr_convention,
        )
        return extrinsics

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
    ) -> "CameraExtrinsics":
        """Return this CameraExtrinsics under a similarity transform of its pose.

        Args:
            scale: Scalar similarity scale factor as a number, numpy array, or torch.Tensor.
            rotation: 3x3 rotation matrix as a numpy array, torch.Tensor, or nested numeric list.
            translation: Length-3 translation as a numpy array, torch.Tensor, tuple, or list.

        Returns:
            A new CameraExtrinsics with the transformed cam2world pose.
        """

        def _validate_inputs() -> None:
            # One factor for the whole pose.
            assert isinstance(scale, (int, float)) or (
                isinstance(scale, (np.ndarray, torch.Tensor)) and scale.shape == ()
            ), (
                "Expected transform scale to be an int, a float, or a numpy array or "
                "torch.Tensor of shape (). "
                f"{type(scale)=} {scale=}"
            )
            validate_rotation_matrix(rotation)
            validate_translation_vector(translation)

        _validate_inputs()

        def _normalize_inputs(
            scale: Union[int, float, np.ndarray, torch.Tensor],
            rotation: Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]],
            translation: Union[
                np.ndarray,
                torch.Tensor,
                Tuple[Union[int, float], Union[int, float], Union[int, float]],
                List[Union[int, float]],
            ],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            scale = torch.as_tensor(scale, device=self._device, dtype=self._dtype)
            assert scale.device == self._device, (
                "Expected the normalized transform scale on the extrinsics device. "
                f"{scale.device=} {self._device=}"
            )
            assert scale.dtype == self._dtype, (
                "Expected the normalized transform scale in the extrinsics dtype. "
                f"{scale.dtype=} {self._dtype=}"
            )
            rotation = torch.as_tensor(rotation, device=self._device, dtype=self._dtype)
            assert rotation.device == self._device, (
                "Expected the normalized transform rotation on the extrinsics device. "
                f"{rotation.device=} {self._device=}"
            )
            assert rotation.dtype == self._dtype, (
                "Expected the normalized transform rotation in the extrinsics dtype. "
                f"{rotation.dtype=} {self._dtype=}"
            )
            translation = torch.as_tensor(
                translation, device=self._device, dtype=self._dtype
            )
            assert translation.device == self._device, (
                "Expected the normalized transform translation on the extrinsics "
                f"device. {translation.device=} {self._device=}"
            )
            assert translation.dtype == self._dtype, (
                "Expected the normalized transform translation in the extrinsics "
                f"dtype. {translation.dtype=} {self._dtype=}"
            )
            return scale, rotation, translation

        scale, rotation, translation = _normalize_inputs(
            scale=scale,
            rotation=rotation,
            translation=translation,
        )

        # The new cam2world rotation is rotation @ R and its translation scale * (rotation @ t) + translation, over the [0, 0, 0, 1] last row the matrix already carries.
        extrinsics_new = torch.cat(
            [
                torch.cat(
                    [
                        rotation @ self._extrinsics[..., :3, :3],
                        scale * (rotation @ self._extrinsics[..., :3, 3:4])
                        + translation.unsqueeze(-1),
                    ],
                    dim=-1,
                ),
                self._extrinsics[..., 3:4, :],
            ],
            dim=-2,
        )
        extrinsics_new[..., :3, :3] = _stabilize_rotation_matrix(
            extrinsics_new[..., :3, :3]
        )

        extrinsics = CameraExtrinsics(
            extrinsics=extrinsics_new,
            extr_convention=self._extr_convention,
        )
        return extrinsics


def _stabilize_rotation_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Project near-orthogonal (..., 3, 3) rotations onto the nearest proper rotations.

    Args:
        rotation: Near-orthogonal ``(..., 3, 3)`` rotations as a float32 or float64 torch.Tensor, the leading dims being the camera batch a single pose leaves empty.

    Returns:
        The nearest proper rotation matrices, in the received shape and dtype.
    """
    # Input validations
    assert rotation.dtype in (torch.float32, torch.float64), (
        "Expected rotation matrix dtype to be float32 or float64. " f"{rotation.dtype=}"
    )

    orthogonality_residual = float(
        torch.max(
            torch.abs(
                rotation @ rotation.transpose(-1, -2)
                - torch.eye(3, dtype=rotation.dtype, device=rotation.device)
            )
        )
    )
    determinant_residual = float(torch.max(torch.abs(torch.linalg.det(rotation) - 1.0)))
    assert (
        max(orthogonality_residual, determinant_residual) <= _ORTHOGONALITY_REPAIR_ATOL
    ), (
        "Expected near-orthogonal rotation matrix before stabilization. "
        f"{orthogonality_residual=} {determinant_residual=} {_ORTHOGONALITY_REPAIR_ATOL=}"
    )

    u, _, v_h = torch.linalg.svd(rotation)
    rotation_fixed = u @ v_h
    signs = torch.cat(
        [
            torch.ones_like(u[..., 0, :2]),
            torch.where(
                torch.linalg.det(rotation_fixed) < 0.0,
                -torch.ones_like(u[..., 0, 2]),
                torch.ones_like(u[..., 0, 2]),
            ).unsqueeze(-1),
        ],
        dim=-1,
    )
    u = u * signs.unsqueeze(-2)
    rotation_fixed = u @ v_h
    validate_rotation_matrix(rotation_fixed)
    return rotation_fixed
