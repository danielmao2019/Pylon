import inspect
import sys
import types
from itertools import product
from pathlib import Path
from typing import Any, List

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

import data.structures.three_d.camera.conventions as convention_module
from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.validation import validate_camera_convention

CONVENTIONS: List[str] = [
    "standard",
    "opengl",
    "opencv",
    "pytorch3d",
    "arkit",
]


def _build_intrinsics() -> torch.Tensor:
    return torch.tensor(
        [
            [400.0, 0.0, 160.0],
            [0.0, 410.0, 120.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _build_extrinsics() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 1.1],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _build_camera(convention: str) -> Any:
    return Camera(
        intrinsics=_build_intrinsics(),
        extrinsics=_build_extrinsics(),
        convention=convention,
        device=torch.device("cpu"),
    )


def _build_cameras(convention: str) -> Any:
    return Cameras(
        intrinsics=[_build_intrinsics()],
        extrinsics=[_build_extrinsics()],
        conventions=[convention],
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("convention", CONVENTIONS)
def test_validate_camera_convention_accepts_all_supported(convention: str) -> None:
    validated = validate_camera_convention(convention=convention)
    assert validated == convention


def test_conventions_module_has_one_main_api_and_eight_helpers() -> None:
    expected_helpers = {
        "_opengl_to_standard",
        "_standard_to_opengl",
        "_opencv_to_standard",
        "_standard_to_opencv",
        "_pytorch3d_to_standard",
        "_standard_to_pytorch3d",
        "_arkit_to_standard",
        "_standard_to_arkit",
    }
    functions = inspect.getmembers(convention_module, inspect.isfunction)
    helper_names = {
        name
        for name, _ in functions
        if name.startswith("_")
        and ("to_standard" in name or name.startswith("_standard_to_"))
    }
    assert helper_names == expected_helpers
    assert hasattr(convention_module, "transform_convention")
    assert not hasattr(convention_module, "_opengl_to_opencv")
    assert not hasattr(convention_module, "_opencv_to_opengl")
    assert not hasattr(convention_module, "_opengl_to_pytorch3d")
    assert not hasattr(convention_module, "_pytorch3d_to_opengl")
    assert not hasattr(convention_module, "_opencv_to_pytorch3d")
    assert not hasattr(convention_module, "_pytorch3d_to_opencv")


@pytest.mark.parametrize(
    "source_convention,target_convention",
    list(product(CONVENTIONS, CONVENTIONS)),
)
def test_camera_conversion_preserves_physical_axes_and_center(
    source_convention: str,
    target_convention: str,
) -> None:
    camera = _build_camera(convention=source_convention)
    converted = camera.to(convention=target_convention)

    assert torch.allclose(
        converted.center,
        camera.center,
        atol=1.0e-06,
        rtol=0.0,
    )
    assert torch.allclose(
        converted.right,
        camera.right,
        atol=1.0e-06,
        rtol=0.0,
    )
    assert torch.allclose(
        converted.forward,
        camera.forward,
        atol=1.0e-06,
        rtol=0.0,
    )
    assert torch.allclose(
        converted.up,
        camera.up,
        atol=1.0e-06,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "source_convention,target_convention",
    list(product(CONVENTIONS, CONVENTIONS)),
)
def test_camera_direct_and_via_standard_conversion_match(
    source_convention: str,
    target_convention: str,
) -> None:
    camera = _build_camera(convention=source_convention)
    converted_direct = camera.to(convention=target_convention)
    converted_via_standard = camera.to(convention="standard").to(
        convention=target_convention
    )

    assert torch.allclose(
        converted_direct.extrinsics,
        converted_via_standard.extrinsics,
        atol=1.0e-06,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "source_convention,target_convention",
    list(product(CONVENTIONS, CONVENTIONS)),
)
def test_camera_round_trip_returns_original_extrinsics(
    source_convention: str,
    target_convention: str,
) -> None:
    camera = _build_camera(convention=source_convention)
    converted = camera.to(convention=target_convention)
    round_trip = converted.to(convention=source_convention)

    assert torch.allclose(
        round_trip.extrinsics,
        camera.extrinsics,
        atol=1.0e-06,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "source_convention,target_convention",
    list(product(CONVENTIONS, CONVENTIONS)),
)
def test_cameras_conversion_preserves_physical_axes_and_center(
    source_convention: str,
    target_convention: str,
) -> None:
    cameras = _build_cameras(convention=source_convention)
    converted = cameras.to(convention=target_convention)

    assert torch.allclose(
        converted.center[0],
        cameras.center[0],
        atol=1.0e-06,
        rtol=0.0,
    )
    assert torch.allclose(
        converted.right[0],
        cameras.right[0],
        atol=1.0e-06,
        rtol=0.0,
    )
    assert torch.allclose(
        converted.forward[0],
        cameras.forward[0],
        atol=1.0e-06,
        rtol=0.0,
    )
    assert torch.allclose(
        converted.up[0],
        cameras.up[0],
        atol=1.0e-06,
        rtol=0.0,
    )
