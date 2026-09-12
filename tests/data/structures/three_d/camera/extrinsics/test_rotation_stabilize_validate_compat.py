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
    r = torch.eye(3, dtype=torch.float16)
    with pytest.raises(AssertionError):
        _stabilize_rotation_matrix(rotation=r)


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
    rotations = torch.stack(
        [
            _random_rotation(dtype=torch.float64, seed=index)
            for index in range(batch_size)
        ]
    )
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
    eps_float64 = torch.finfo(torch.float64).eps
    eps_float32 = torch.finfo(torch.float32).eps
    a = 5e-7
    assert a > 32 * eps_float64, (
        "Expected the deviation to sit above the float64 tolerance. "
        f"{a=} {32 * eps_float64=}"
    )
    assert a < 32 * eps_float32, (
        "Expected the deviation to sit below the float32 tolerance. "
        f"{a=} {32 * eps_float32=}"
    )

    m = torch.eye(3, dtype=torch.float64)
    m[0, 0] = 1.0 + a

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

    reflection = rotation.clone()
    reflection[:, 0] = -reflection[:, 0]
    assert float(torch.linalg.det(reflection)) < 0.0, (
        "Expected negating one column of a proper rotation to give an "
        f"orthonormal matrix of determinant -1. {float(torch.linalg.det(reflection))=}"
    )
    with pytest.raises(AssertionError):
        validate_rotation_matrix(obj=reflection)

    extrinsics = torch.eye(4, dtype=torch.float64)
    extrinsics[:3, :3] = reflection
    batch = torch.stack([extrinsics, extrinsics])
    with pytest.raises(AssertionError):
        validate_camera_extrinsics(obj=batch)


def _random_rotation(dtype: torch.dtype, seed: int) -> torch.Tensor:
    """One reproducible proper rotation per seed, so every rotation-building test starts from a real rotation instead of a hand-written matrix.

    Args:
        dtype: Floating torch dtype the rotation is returned in.
        seed: Seed of the generator the random matrix is drawn from.

    Returns:
        A ``(3, 3)`` proper rotation torch.Tensor of determinant +1, in ``dtype``.
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q @ torch.diag(torch.sign(torch.diagonal(r)))
    if float(torch.linalg.det(q)) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(dtype)
