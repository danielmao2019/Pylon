import pytest
import torch

from data.structures.three_d.camera.extrinsics.camera_extrinsics import (
    _stabilize_rotation_matrix,
)
from data.structures.three_d.camera.extrinsics.validation import (
    validate_camera_extrinsics,
    validate_rotation_matrix,
)


def test_stabilize_accepts_float32_and_float64() -> None:
    """_stabilize_rotation_matrix accepts a float32 or float64 near-orthogonal rotation, returns the same dtype, and its output passes validate_rotation_matrix.

    Args:
        None.

    Returns:
        None.
    """
    for dtype in (torch.float32, torch.float64):
        r = _random_rotation(dtype=dtype, seed=1) @ _random_rotation(
            dtype=dtype, seed=2
        )
        out = _stabilize_rotation_matrix(rotation=r)
        assert out.dtype == dtype, (
            "Expected the stabilized rotation to keep the dtype it received. "
            f"{out.dtype=} {dtype=}"
        )
        validate_rotation_matrix(obj=out)


def test_stabilize_rejects_unsupported_dtype() -> None:
    """_stabilize_rotation_matrix raises on a dtype outside {float32, float64} (e.g. float16).

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(AssertionError):
        _stabilize_rotation_matrix(rotation=torch.eye(3, dtype=torch.float16))


def test_stabilized_batch_passes_validator() -> None:
    """A [B, 3, 3] batch stabilized in one call matches stabilizing each rotation alone, and the cam2world batch it builds passes validate_camera_extrinsics for both float32 and float64.

    Args:
        None.

    Returns:
        None.
    """
    batch_size = 32
    for dtype in (torch.float32, torch.float64):
        rotations = torch.stack(
            [
                _random_rotation(dtype=dtype, seed=index)
                @ _random_rotation(dtype=dtype, seed=index + 5000)
                for index in range(batch_size)
            ]
        )
        stabilized = _stabilize_rotation_matrix(rotation=rotations)

        for index, rotation in enumerate(rotations):
            stabilized_alone = _stabilize_rotation_matrix(rotation=rotation)
            assert torch.equal(stabilized[index], stabilized_alone), (
                "Expected stabilizing a batch in one call to match stabilizing each "
                f"rotation of that batch alone. {dtype=} {index=} {stabilized[index]=} {stabilized_alone=}"
            )

        extrinsics = torch.eye(4, dtype=dtype).repeat(batch_size, 1, 1)
        extrinsics[:, :3, :3] = stabilized
        validate_camera_extrinsics(obj=extrinsics)


def test_stabilize_rejects_a_reflection() -> None:
    """A batch mixing proper rotations with reflections is refused rather than sign-repaired, since a camera's rotation is proper by construction.

    Args:
        None.

    Returns:
        None.
    """
    batch_size = 4
    rotation_list = []
    for index in range(batch_size):
        rotation_list.append(_random_rotation(dtype=torch.float64, seed=index))
    rotations = torch.stack(rotation_list)
    rotations[1::2, :, 0] = -rotations[1::2, :, 0]

    with pytest.raises(AssertionError):
        _stabilize_rotation_matrix(rotation=rotations)


def test_validator_threshold_is_dtype_aware() -> None:
    """A fixed near-orthogonality deviation between the float64 and float32 tolerances passes validate_rotation_matrix as float32 but is rejected as float64.

    Args:
        None.

    Returns:
        None.
    """
    m = torch.diag(torch.tensor([1.0 + 5e-7, 1.0, 1.0], dtype=torch.float64))

    validate_rotation_matrix(obj=m.to(torch.float32))

    with pytest.raises(AssertionError):
        validate_rotation_matrix(obj=m)


def test_validator_requires_determinant_plus_one() -> None:
    """A camera's rotation is validated to have determinant +1, so a reflection is inexpressible as camera extrinsics and any change of handedness has to be carried by the geometry instead.

    Args:
        None.

    Returns:
        None.
    """
    rotation = _random_rotation(dtype=torch.float64, seed=7)
    validate_rotation_matrix(obj=rotation)

    reflection = rotation * torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    with pytest.raises(AssertionError):
        validate_rotation_matrix(obj=reflection)

    with pytest.raises(AssertionError):
        validate_camera_extrinsics(
            obj=torch.block_diag(reflection, torch.ones(1, 1, dtype=torch.float64))[
                None
            ]
        )


def _random_rotation(dtype: torch.dtype, seed: int) -> torch.Tensor:
    """One reproducible proper rotation per seed, so every rotation-building test starts from a real rotation instead of a hand-written matrix.

    Args:
        dtype: Floating torch dtype the rotation is returned in.
        seed: Seed of the generator the random matrix is drawn from.

    Returns:
        A ``(3, 3)`` proper rotation torch.Tensor of determinant +1, in ``dtype``.
    """
    a = torch.randn(
        3, 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64
    )
    q, r = torch.linalg.qr(a)
    q = q @ torch.diag(torch.sign(torch.diagonal(r)))
    if float(torch.linalg.det(q)) < 0:
        q[:, 0] = -q[:, 0]
    q = q.to(dtype)
    return q
