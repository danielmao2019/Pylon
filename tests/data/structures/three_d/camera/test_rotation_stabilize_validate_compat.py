import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


def _install_namespace_package(package_name: str, package_path: Path) -> None:
    if package_name in sys.modules:
        return

    module = types.ModuleType(package_name)
    module.__file__ = str(package_path / "__init__.py")
    module.__path__ = [str(package_path)]
    sys.modules[package_name] = module


REPO_ROOT = Path(__file__).resolve().parents[5]
_install_namespace_package(package_name="utils", package_path=REPO_ROOT / "utils")
_install_namespace_package(
    package_name="utils.ops", package_path=REPO_ROOT / "utils" / "ops"
)
_install_namespace_package(package_name="data", package_path=REPO_ROOT / "data")
_install_namespace_package(
    package_name="data.structures", package_path=REPO_ROOT / "data" / "structures"
)
_install_namespace_package(
    package_name="data.structures.three_d",
    package_path=REPO_ROOT / "data" / "structures" / "three_d",
)
_install_namespace_package(
    package_name="data.structures.three_d.camera",
    package_path=REPO_ROOT / "data" / "structures" / "three_d" / "camera",
)

from data.structures.three_d.camera.camera import _stabilize_rotation_matrix
from data.structures.three_d.camera.validation import (
    validate_camera_extrinsics,
    validate_rotation_matrix,
)


def _random_rotation(dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q @ torch.diag(torch.sign(torch.diagonal(r)))
    if float(torch.linalg.det(q)) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stabilize_accepts_float32_and_float64(dtype: torch.dtype) -> None:
    r = _random_rotation(dtype, 1) @ _random_rotation(dtype, 2)
    out = _stabilize_rotation_matrix(r)
    assert out.dtype == dtype
    validate_rotation_matrix(out)


def test_stabilize_rejects_unsupported_dtype() -> None:
    r = torch.eye(3, dtype=torch.float16)
    with pytest.raises(AssertionError):
        _stabilize_rotation_matrix(r)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stabilized_batch_passes_validator(dtype: torch.dtype) -> None:
    extrinsics_list = []
    for index in range(200):
        extrinsics = torch.eye(4, dtype=dtype)
        rotation = _stabilize_rotation_matrix(
            _random_rotation(dtype, index) @ _random_rotation(dtype, index + 5000)
        )
        extrinsics[:3, :3] = rotation
        extrinsics_list.append(extrinsics)
    batch = torch.stack(extrinsics_list)
    assert batch.shape == (200, 4, 4)
    validate_camera_extrinsics(batch)


def test_validator_threshold_is_dtype_aware() -> None:
    eps_float64 = float(np.finfo(np.float64).eps)
    eps_float32 = float(np.finfo(np.float32).eps)
    a = 5e-7
    assert a > 32 * eps_float64
    assert a < 32 * eps_float32

    m = torch.eye(3, dtype=torch.float64)
    m[0, 0] = 1.0 + a

    validate_rotation_matrix(m.to(torch.float32))

    with pytest.raises(AssertionError):
        validate_rotation_matrix(m)
