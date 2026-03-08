"""Coordinate system transformation utilities for point cloud rendering.

This module provides utilities to handle different camera coordinate system conventions
and transform them to standardized coordinate systems used in point cloud rendering.
"""

from typing import TYPE_CHECKING

import torch

from data.structures.three_d.camera.validation import validate_camera_convention
from utils.ops.materialize_tensor import materialize_tensor

if TYPE_CHECKING:
    from data.structures.three_d.camera.camera import Camera


def transform_convention(
    camera: "Camera",
    target_convention: str = "standard",
) -> torch.Tensor:
    """Transform camera pose between different coordinate system conventions.

    Supported coordinate conventions:
    - "standard": X=right, Y=forward, Z=up (camera looks down +Y axis)
    - "opengl": X=right, Y=up, Z=backward (camera looks down -Z axis)
    - "opencv": X=right, Y=down, Z=forward (camera looks down +Z axis)
    - "pytorch3d": X=left, Y=up, Z=forward (right-handed; camera looks down +Z axis)
    - "arkit": X=down, Y=left, Z=forward

    Args:
        camera: Camera containing extrinsics/convention
        target_convention: Target coordinate convention ("standard", "opengl", "opencv", "pytorch3d", "arkit")

    Returns:
        Transformed 4x4 camera pose in target convention

    Raises:
        AssertionError: If unknown coordinate convention is specified
    """
    # Input validations
    validate_camera_convention(camera.convention)
    validate_camera_convention(target_convention)

    source_convention = camera.convention
    extrinsics = camera.extrinsics

    if source_convention == target_convention:
        return extrinsics

    source_to_standard = torch.eye(4, dtype=torch.float32)
    if source_convention == "opengl":
        source_to_standard = _opengl_to_standard()
    elif source_convention == "opencv":
        source_to_standard = _opencv_to_standard()
    elif source_convention == "pytorch3d":
        source_to_standard = _pytorch3d_to_standard()
    elif source_convention == "arkit":
        source_to_standard = _arkit_to_standard()
    else:
        assert (
            source_convention == "standard"
        ), f"Unsupported convention: {source_convention}"

    standard_to_target = torch.eye(4, dtype=torch.float32)
    if target_convention == "opengl":
        standard_to_target = _standard_to_opengl()
    elif target_convention == "opencv":
        standard_to_target = _standard_to_opencv()
    elif target_convention == "pytorch3d":
        standard_to_target = _standard_to_pytorch3d()
    elif target_convention == "arkit":
        standard_to_target = _standard_to_arkit()
    else:
        assert (
            target_convention == "standard"
        ), f"Unsupported convention: {target_convention}"

    # Compose conversion strictly through standard:
    # source -> standard -> target.
    source_to_target = standard_to_target @ source_to_standard

    # Move transform to same device and dtype as camera pose.
    if (
        extrinsics.device != source_to_target.device
        or extrinsics.dtype != source_to_target.dtype
    ):
        source_to_target = source_to_target.to(
            device=extrinsics.device,
            dtype=extrinsics.dtype,
        )

    camera_extrinsics_transformed = extrinsics @ torch.inverse(
        materialize_tensor(source_to_target)
    )

    return camera_extrinsics_transformed


def _opengl_to_standard() -> torch.Tensor:
    """Get transformation matrix from OpenGL to standard coordinate system.

    OpenGL convention:
    - X: right
    - Y: up
    - Z: backward (camera looks down -Z axis)

    Standard convention:
    - X: right
    - Y: forward (camera looks down +Y axis)
    - Z: up

    Mapping summary:
    - OpenGL right (+X) -> Standard right (+X)
    - OpenGL up (+Y) -> Standard up (+Z)
    - OpenGL forward (-Z) -> Standard forward (+Y)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [1, 0, 0, 0],  # Keep X: OpenGL right -> Standard right
            [0, 0, -1, 0],  # -Z->Y: OpenGL forward -> Standard forward
            [0, 1, 0, 0],  # Y->Z: OpenGL up -> Standard up
            [0, 0, 0, 1],  # Homogeneous
        ],
        dtype=torch.float32,
    )


def _standard_to_opengl() -> torch.Tensor:
    """Get transformation matrix from standard to OpenGL coordinate system.

    Standard convention:
    - X: right
    - Y: forward
    - Z: up

    OpenGL convention:
    - X: right
    - Y: up
    - Z: backward (camera looks down -Z axis)

    Mapping summary:
    - Standard right (+X) -> OpenGL right (+X)
    - Standard forward (+Y) -> OpenGL forward (-Z)
    - Standard up (+Z) -> OpenGL up (+Y)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [1, 0, 0, 0],  # Keep X: Standard right -> OpenGL right
            [0, 0, 1, 0],  # Z->Y: Standard up -> OpenGL up
            [0, -1, 0, 0],  # Y->-Z: Standard forward -> OpenGL forward
            [0, 0, 0, 1],  # Homogeneous
        ],
        dtype=torch.float32,
    )


def _opencv_to_standard() -> torch.Tensor:
    """Get transformation matrix from OpenCV to standard coordinate system.

    OpenCV convention:
    - X: right
    - Y: down
    - Z: forward (camera looks down +Z axis)

    Standard convention:
    - X: right
    - Y: forward (camera looks down +Y axis)
    - Z: up

    Mapping summary:
    - OpenCV right (+X) -> Standard right (+X)
    - OpenCV forward (+Z) -> Standard forward (+Y)
    - OpenCV down (+Y) -> Standard down (-Z)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [1, 0, 0, 0],  # Keep X: OpenCV right -> Standard right
            [0, 0, 1, 0],  # Z->Y: OpenCV forward -> Standard forward
            [0, -1, 0, 0],  # -Y->Z: OpenCV down -> Standard down
            [0, 0, 0, 1],  # Homogeneous
        ],
        dtype=torch.float32,
    )


def _standard_to_opencv() -> torch.Tensor:
    """Get transformation matrix from standard to OpenCV coordinate system.

    Standard convention:
    - X: right
    - Y: forward
    - Z: up

    OpenCV convention:
    - X: right
    - Y: down
    - Z: forward (camera looks down +Z axis)

    Mapping summary:
    - Standard right (+X) -> OpenCV right (+X)
    - Standard forward (+Y) -> OpenCV forward (+Z)
    - Standard up (+Z) -> OpenCV down (+Y)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [1, 0, 0, 0],  # Keep X: Standard right -> OpenCV right
            [0, 0, -1, 0],  # Z->Y: Standard up -> OpenCV down
            [0, 1, 0, 0],  # Y->Z: Standard forward -> OpenCV forward
            [0, 0, 0, 1],  # Homogeneous
        ],
        dtype=torch.float32,
    )


def _pytorch3d_to_standard() -> torch.Tensor:
    """Get transformation matrix from PyTorch3D to standard coordinate system.

    PyTorch3D convention (right-handed):
    - X: left (+X points left)
    - Y: up (+Y points up)
    - Z: forward (+Z points from us to scene, out from image plane)

    Standard convention:
    - X: right
    - Y: forward
    - Z: up

    Mapping summary:
    - PyTorch3D left (+X) -> Standard left (-X)
    - PyTorch3D up (+Y) -> Standard up (+Z)
    - PyTorch3D forward (+Z) -> Standard forward (+Y)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [-1, 0, 0, 0],  # Negate X: PyTorch3D left -> Standard right
            [0, 0, 1, 0],  # Z->Y: PyTorch3D forward -> Standard forward
            [0, 1, 0, 0],  # Y->Z: PyTorch3D up -> Standard up
            [0, 0, 0, 1],  # Homogeneous
        ],
        dtype=torch.float32,
    )


def _standard_to_pytorch3d() -> torch.Tensor:
    """Get transformation matrix from standard to PyTorch3D coordinate system.

    Standard convention:
    - X: right
    - Y: forward
    - Z: up

    PyTorch3D convention (right-handed):
    - X: left (+X points left)
    - Y: up (+Y points up)
    - Z: forward (+Z points from us to scene, out from image plane)

    Mapping summary:
    - Standard right (+X) -> PyTorch3D right (-X)
    - Standard forward (+Y) -> PyTorch3D forward (+Z)
    - Standard up (+Z) -> PyTorch3D up (+Y)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [-1, 0, 0, 0],  # Negate X: Standard right -> PyTorch3D left
            [0, 0, 1, 0],  # Z->Y: Standard up -> PyTorch3D up
            [0, 1, 0, 0],  # Y->Z: Standard forward -> PyTorch3D forward
            [0, 0, 0, 1],  # Homogeneous
        ],
        dtype=torch.float32,
    )


def _arkit_to_standard() -> torch.Tensor:
    """Get transformation matrix from ARKit to standard coordinate system.

    ARKit convention:
    - X: down
    - Y: left
    - Z: forward

    Standard convention:
    - X: right
    - Y: forward
    - Z: up

    Mapping summary:
    - ARKit down (+X) -> Standard down (-Z)
    - ARKit left (+Y) -> Standard left (-X)
    - ARKit forward (+Z) -> Standard forward (+Y)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [0, -1, 0, 0],  # -Y->X: ARKit left -> Standard left
            [0, 0, 1, 0],  # Z->Y: ARKit forward -> Standard forward
            [-1, 0, 0, 0],  # -X->Z: ARKit down -> Standard down
            [0, 0, 0, 1],  # Homogeneous
        ],
        dtype=torch.float32,
    )


def _standard_to_arkit() -> torch.Tensor:
    """Get transformation matrix from standard to ARKit coordinate system.

    Standard convention:
    - X: right
    - Y: forward
    - Z: up

    ARKit convention:
    - X: down
    - Y: left
    - Z: forward

    Mapping summary:
    - Standard right (+X) -> ARKit right (-Y)
    - Standard forward (+Y) -> ARKit forward (+Z)
    - Standard up (+Z) -> ARKit up (-X)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [0, 0, -1, 0],  # -Z->X: Standard up -> ARKit up
            [-1, 0, 0, 0],  # -X->Y: Standard right -> ARKit right
            [0, 1, 0, 0],  # Y->Z: Standard forward -> ARKit forward
            [0, 0, 0, 1],  # Homogeneous
        ],
        dtype=torch.float32,
    )
