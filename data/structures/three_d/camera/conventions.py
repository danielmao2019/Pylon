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

    Args:
        camera: Camera containing extrinsics/convention
        target_convention: Target coordinate convention ("opengl", "standard", "opencv", "pytorch3d")

    Returns:
        Transformed 4x4 camera pose in target convention

    Raises:
        AssertionError: If unknown coordinate convention is specified
    """
    # Input validations
    validate_camera_convention(camera.convention)
    validate_camera_convention(target_convention)

    extrinsics = camera.extrinsics
    source_convention = camera.convention

    if source_convention == target_convention:
        return extrinsics

    # Get appropriate transformation matrix
    if source_convention == "opengl" and target_convention == "standard":
        transform = _opengl_to_standard()
    elif source_convention == "standard" and target_convention == "opengl":
        transform = _standard_to_opengl()
    elif source_convention == "opengl" and target_convention == "opencv":
        transform = _opengl_to_opencv()
    elif source_convention == "opencv" and target_convention == "opengl":
        transform = _opencv_to_opengl()
    elif source_convention == "opencv" and target_convention == "standard":
        transform = _opencv_to_standard()
    elif source_convention == "standard" and target_convention == "opencv":
        transform = _standard_to_opencv()
    elif source_convention == "opengl" and target_convention == "pytorch3d":
        transform = _opengl_to_pytorch3d()
    elif source_convention == "pytorch3d" and target_convention == "opengl":
        transform = _pytorch3d_to_opengl()
    elif source_convention == "standard" and target_convention == "pytorch3d":
        transform = _standard_to_pytorch3d()
    elif source_convention == "pytorch3d" and target_convention == "standard":
        transform = _pytorch3d_to_standard()
    elif source_convention == "opencv" and target_convention == "pytorch3d":
        transform = _opencv_to_pytorch3d()
    elif source_convention == "pytorch3d" and target_convention == "opencv":
        transform = _pytorch3d_to_opencv()
    else:
        assert False, (
            f"Unsupported transformation: {source_convention} -> {target_convention}. "
            f"Supported: 'opengl' <-> 'standard', 'opengl' <-> 'opencv', "
            f"'opencv' <-> 'standard', 'pytorch3d' <-> 'opengl', 'pytorch3d' <-> 'standard', "
            f"'pytorch3d' <-> 'opencv'"
        )

    # Move transform to same device and dtype as camera pose
    if extrinsics.device != transform.device or extrinsics.dtype != transform.dtype:
        transform = transform.to(device=extrinsics.device, dtype=extrinsics.dtype)

    camera_extrinsics_transformed = extrinsics @ torch.inverse(
        materialize_tensor(transform)
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
    - OpenGL right (+X) → Standard right (+X)
    - OpenGL up (+Y) → Standard up (+Z)
    - OpenGL forward (-Z) → Standard forward (+Y)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [1, 0, 0, 0],  # Keep X: OpenGL right → Standard right
            [0, 0, -1, 0],  # -Z→Y: OpenGL forward → Standard forward
            [0, 1, 0, 0],  # Y→Z: OpenGL up → Standard up
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
    - Standard right (+X) → OpenGL right (+X)
    - Standard forward (+Y) → OpenGL forward (-Z)
    - Standard up (+Z) → OpenGL up (+Y)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [1, 0, 0, 0],  # Keep X: Standard right → OpenGL right
            [0, 0, 1, 0],  # Z→Y: Standard up → OpenGL up
            [0, -1, 0, 0],  # Y→-Z: Standard forward → OpenGL forward
            [0, 0, 0, 1],  # Homogeneous
        ],
        dtype=torch.float32,
    )


def _opengl_to_opencv() -> torch.Tensor:
    """Get transformation matrix from OpenGL to OpenCV coordinate system.

    OpenGL convention:
    - X: right
    - Y: up
    - Z: backward (camera looks down -Z axis)

    OpenCV convention:
    - X: right
    - Y: down
    - Z: forward (camera looks down +Z axis)

    Mapping summary:
    - OpenGL right (+X) → OpenCV right (+X)
    - OpenGL up (+Y) → OpenCV down (+Y)
    - OpenGL forward (-Z) → OpenCV forward (+Z)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [1, 0, 0, 0],  # Keep X: OpenGL right → OpenCV right
            [0, -1, 0, 0],  # Negate Y: OpenGL up → OpenCV down
            [0, 0, -1, 0],  # Negate Z: OpenGL forward → OpenCV forward
            [0, 0, 0, 1],  # Homogeneous
        ],
        dtype=torch.float32,
    )


def _opencv_to_opengl() -> torch.Tensor:
    """Get transformation matrix from OpenCV to OpenGL coordinate system.

    OpenCV convention:
    - X: right
    - Y: down
    - Z: forward (camera looks down +Z axis)

    OpenGL convention:
    - X: right
    - Y: up
    - Z: backward (camera looks down -Z axis)

    Mapping summary:
    - OpenCV right (+X) → OpenGL right (+X)
    - OpenCV down (+Y) → OpenGL up (+Y)
    - OpenCV forward (+Z) → OpenGL forward (-Z)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [1, 0, 0, 0],  # Keep X: OpenCV right → OpenGL right
            [0, -1, 0, 0],  # Negate Y: OpenCV down → OpenGL up
            [0, 0, -1, 0],  # Negate Z: OpenCV forward → OpenGL forward
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
    - OpenCV right (+X) → Standard right (+X)
    - OpenCV forward (+Z) → Standard forward (+Y)
    - OpenCV down (+Y) → Standard down (-Z)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [1, 0, 0, 0],  # Keep X: OpenCV right → Standard right
            [0, 0, 1, 0],  # Z→Y: OpenCV forward → Standard forward
            [0, -1, 0, 0],  # -Y→Z: OpenCV down → Standard down
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
    - Standard right (+X) → OpenCV right (+X)
    - Standard forward (+Y) → OpenCV forward (+Z)
    - Standard up (+Z) → OpenCV down (+Y)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [1, 0, 0, 0],  # Keep X: Standard right → OpenCV right
            [0, 0, -1, 0],  # Z→Y: Standard up → OpenCV down
            [0, 1, 0, 0],  # Y→Z: Standard forward → OpenCV forward
            [0, 0, 0, 1],  # Homogeneous
        ],
        dtype=torch.float32,
    )


def _opengl_to_pytorch3d() -> torch.Tensor:
    """Get transformation matrix from OpenGL to PyTorch3D coordinate system.

    OpenGL convention:
    - X: right
    - Y: up
    - Z: backward (camera looks down -Z axis)

    PyTorch3D convention (right-handed):
    - X: left (+X points left)
    - Y: up (+Y points up)
    - Z: forward (+Z points from us to scene, out from image plane)

    Mapping summary:
    - OpenGL right (+X) → PyTorch3D right (-X)
    - OpenGL up (+Y) → PyTorch3D up (+Y)
    - OpenGL forward (-Z) → PyTorch3D forward (+Z)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [-1, 0, 0, 0],  # Negate X: OpenGL right → PyTorch3D left
            [0, 1, 0, 0],  # Keep Y: OpenGL up → PyTorch3D up
            [0, 0, -1, 0],  # Negate Z: OpenGL backward → PyTorch3D forward
            [0, 0, 0, 1],  # Homogeneous
        ],
        dtype=torch.float32,
    )


def _pytorch3d_to_opengl() -> torch.Tensor:
    """Get transformation matrix from PyTorch3D to OpenGL coordinate system."""
    return _opengl_to_pytorch3d()


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
    - Standard right (+X) → PyTorch3D right (-X)
    - Standard forward (+Y) → PyTorch3D forward (+Z)
    - Standard up (+Z) → PyTorch3D up (+Y)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [-1, 0, 0, 0],  # Negate X: Standard right → PyTorch3D left
            [0, 0, 1, 0],  # Z→Y: Standard up → PyTorch3D up
            [0, 1, 0, 0],  # Y→Z: Standard forward → PyTorch3D forward
            [0, 0, 0, 1],  # Homogeneous
        ],
        dtype=torch.float32,
    )


def _pytorch3d_to_standard() -> torch.Tensor:
    """Get transformation matrix from PyTorch3D to standard coordinate system."""
    return _standard_to_pytorch3d()


def _opencv_to_pytorch3d() -> torch.Tensor:
    """Get transformation matrix from OpenCV to PyTorch3D coordinate system.

    OpenCV convention:
    - X: right
    - Y: down
    - Z: forward

    PyTorch3D convention (right-handed):
    - X: left (+X points left)
    - Y: up (+Y points up)
    - Z: forward (+Z points from us to scene, out from image plane)

    Mapping summary:
    - OpenCV right (+X) → PyTorch3D right (-X)
    - OpenCV down (+Y) → PyTorch3D down (-Y)
    - OpenCV forward (+Z) → PyTorch3D forward (+Z)

    Returns:
        4x4 transformation matrix
    """
    return torch.tensor(
        [
            [-1, 0, 0, 0],  # Negate X: OpenCV right → PyTorch3D left
            [0, -1, 0, 0],  # Negate Y: OpenCV down → PyTorch3D up
            [0, 0, 1, 0],  # Keep Z: OpenCV forward → PyTorch3D forward
            [0, 0, 0, 1],  # Homogeneous
        ],
        dtype=torch.float32,
    )


def _pytorch3d_to_opencv() -> torch.Tensor:
    """Get transformation matrix from PyTorch3D to OpenCV coordinate system."""
    return _opencv_to_pytorch3d()
