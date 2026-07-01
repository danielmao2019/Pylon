import inspect
from itertools import product
from typing import List

import pytest
import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.extrinsics import conventions as conventions_module
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.extrinsics.validation import (
    validate_camera_convention,
)
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    CameraIntrinsicsPinhole,
)

CONVENTIONS: List[str] = [
    "standard",
    "opengl",
    "opencv",
    "pytorch3d",
    "arkit",
]


def _build_extrinsics_matrix() -> torch.Tensor:
    """Build a valid 4x4 cam2world matrix with a proper rotation.

    Args:
        None.

    Returns:
        A 4x4 float32 camera-to-world matrix whose 3x3 block is a proper rotation.
    """
    return torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 1.1],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _build_intrinsics() -> CameraIntrinsicsPinhole:
    """Build a pinhole CameraIntrinsics fixture.

    Args:
        None.

    Returns:
        A CameraIntrinsicsPinhole on the CPU.
    """
    return CameraIntrinsicsPinhole(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0},
        device="cpu",
    )


def _build_extrinsics(convention: str) -> CameraExtrinsics:
    """Build a CameraExtrinsics fixture in the given convention.

    Args:
        convention: Coordinate-frame convention string.

    Returns:
        A CameraExtrinsics on the CPU in the given convention.
    """
    return CameraExtrinsics(
        extrinsics=_build_extrinsics_matrix(),
        convention=convention,
        device="cpu",
    )


def _build_cameras(convention: str) -> Cameras:
    """Build a one-camera Cameras fixture in the given convention.

    Args:
        convention: Coordinate-frame convention string.

    Returns:
        A Cameras with one camera on the CPU in the given convention.
    """
    return Cameras(
        intrinsics=[_build_intrinsics()],
        extrinsics=[_build_extrinsics(convention=convention)],
        device="cpu",
    )


@pytest.mark.parametrize("convention", CONVENTIONS)
def test_validate_camera_convention_accepts_all_supported(convention: str) -> None:
    """validate_camera_convention accepts every supported convention string.

    Args:
        convention: The convention string under test.

    Returns:
        None.
    """
    assert validate_camera_convention(convention) == convention, f"{convention=}"


def test_conventions_module_has_one_main_api_and_eight_helpers() -> None:
    """The relocated extrinsics conventions module has one main API and eight helpers.

    Args:
        None.

    Returns:
        None.
    """
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
    functions = inspect.getmembers(conventions_module, inspect.isfunction)
    helper_names = {
        name
        for name, _ in functions
        if name.startswith("_")
        and ("to_standard" in name or name.startswith("_standard_to_"))
    }
    assert helper_names == expected_helpers, f"{helper_names=}"
    assert hasattr(conventions_module, "transform_convention"), f"{functions=}"
    assert not hasattr(conventions_module, "_opengl_to_opencv"), f"{functions=}"
    assert not hasattr(conventions_module, "_opencv_to_pytorch3d"), f"{functions=}"


@pytest.mark.parametrize(
    "source_convention,target_convention",
    list(product(CONVENTIONS, CONVENTIONS)),
)
def test_extrinsics_conversion_preserves_physical_axes_and_center(
    source_convention: str,
    target_convention: str,
) -> None:
    """Converting a CameraExtrinsics preserves its physical axes and center.

    Args:
        source_convention: Source coordinate-frame convention.
        target_convention: Target coordinate-frame convention.

    Returns:
        None.
    """
    extrinsics = _build_extrinsics(convention=source_convention)
    converted = extrinsics.to(convention=target_convention)
    assert torch.allclose(
        converted.center, extrinsics.center, atol=1.0e-06, rtol=0.0
    ), f"{converted.center=} {extrinsics.center=}"
    assert torch.allclose(
        converted.right, extrinsics.right, atol=1.0e-06, rtol=0.0
    ), f"{converted.right=} {extrinsics.right=}"
    assert torch.allclose(
        converted.forward, extrinsics.forward, atol=1.0e-06, rtol=0.0
    ), f"{converted.forward=} {extrinsics.forward=}"
    assert torch.allclose(
        converted.up, extrinsics.up, atol=1.0e-06, rtol=0.0
    ), f"{converted.up=} {extrinsics.up=}"


@pytest.mark.parametrize(
    "source_convention,target_convention",
    list(product(CONVENTIONS, CONVENTIONS)),
)
def test_extrinsics_direct_and_via_standard_conversion_match(
    source_convention: str,
    target_convention: str,
) -> None:
    """Direct conversion matches conversion routed through the standard convention.

    Args:
        source_convention: Source coordinate-frame convention.
        target_convention: Target coordinate-frame convention.

    Returns:
        None.
    """
    extrinsics = _build_extrinsics(convention=source_convention)
    converted_direct = extrinsics.to(convention=target_convention)
    converted_via_standard = extrinsics.to(convention="standard").to(
        convention=target_convention
    )
    assert torch.allclose(
        converted_direct.extrinsics,
        converted_via_standard.extrinsics,
        atol=1.0e-06,
        rtol=0.0,
    ), f"{converted_direct.extrinsics=} {converted_via_standard.extrinsics=}"


@pytest.mark.parametrize(
    "source_convention,target_convention",
    list(product(CONVENTIONS, CONVENTIONS)),
)
def test_extrinsics_round_trip_returns_original_matrix(
    source_convention: str,
    target_convention: str,
) -> None:
    """Converting a CameraExtrinsics to another convention and back is identity.

    Args:
        source_convention: Source coordinate-frame convention.
        target_convention: Target coordinate-frame convention.

    Returns:
        None.
    """
    extrinsics = _build_extrinsics(convention=source_convention)
    round_trip = extrinsics.to(convention=target_convention).to(
        convention=source_convention
    )
    assert torch.allclose(
        round_trip.extrinsics, extrinsics.extrinsics, atol=1.0e-06, rtol=0.0
    ), f"{round_trip.extrinsics=} {extrinsics.extrinsics=}"


@pytest.mark.parametrize("convention", CONVENTIONS)
def test_extrinsics_w2c_is_inverse_of_extrinsics(convention: str) -> None:
    """CameraExtrinsics.w2c is the inverse of the 4x4 cam2world matrix.

    Args:
        convention: The convention string under test.

    Returns:
        None.
    """
    extrinsics = _build_extrinsics(convention=convention)
    product_matrix = extrinsics.w2c @ extrinsics.extrinsics
    identity = torch.eye(4, dtype=extrinsics.extrinsics.dtype)
    assert torch.allclose(
        product_matrix, identity, atol=1.0e-05, rtol=0.0
    ), f"{product_matrix=}"


@pytest.mark.parametrize(
    "source_convention,target_convention",
    list(product(CONVENTIONS, CONVENTIONS)),
)
def test_cameras_conversion_preserves_physical_axes_and_center(
    source_convention: str,
    target_convention: str,
) -> None:
    """Converting a Cameras collection preserves each camera's axes and center.

    Args:
        source_convention: Source coordinate-frame convention.
        target_convention: Target coordinate-frame convention.

    Returns:
        None.
    """
    cameras = _build_cameras(convention=source_convention)
    converted = cameras.to(convention=target_convention)
    assert torch.allclose(
        converted.center[0], cameras.center[0], atol=1.0e-06, rtol=0.0
    ), f"{converted.center=} {cameras.center=}"
    assert torch.allclose(
        converted.right[0], cameras.right[0], atol=1.0e-06, rtol=0.0
    ), f"{converted.right=} {cameras.right=}"
    assert torch.allclose(
        converted.forward[0], cameras.forward[0], atol=1.0e-06, rtol=0.0
    ), f"{converted.forward=} {cameras.forward=}"
    assert torch.allclose(
        converted.up[0], cameras.up[0], atol=1.0e-06, rtol=0.0
    ), f"{converted.up=} {cameras.up=}"
