"""Coordinate system transformation utilities for point cloud rendering.

This module provides utilities to handle different camera coordinate system conventions
and transform them to standardized coordinate systems used in point cloud rendering.
"""

import torch
from utils.ops.materialize_tensor import materialize_tensor


def _opengl_to_standard() -> torch.Tensor:
    """Get transformation matrix from OpenGL to standard coordinate system.
    
    OpenGL convention (used by rendering):
    - X: right
    - Y: up  
    - Z: backward (camera looks down -Z axis)
    
    Standard convention (used by crop module):
    - X: right
    - Y: forward (camera looks down +Y axis)
    - Z: up
    
    The transformation maps coordinate axes:
    - OpenGL X (right) -> Standard X (right)
    - OpenGL Y (up) -> Standard Z (up)  
    - OpenGL Z (backward) -> Standard -Y (backward)
    
    This preserves the camera's orientation in world space:
    - OpenGL forward (-Z) -> Standard forward (+Y)
    - OpenGL up (+Y) -> Standard up (+Z)
    - OpenGL right (+X) -> Standard right (+X)
    
    Returns:
        4x4 transformation matrix
    """
    return torch.tensor([
        [1, 0, 0, 0],    # Standard X = OpenGL X (right stays right)
        [0, 0, -1, 0],   # Standard Y = OpenGL -Z (forward: -Z becomes +Y)  
        [0, 1, 0, 0],    # Standard Z = OpenGL Y (up: Y becomes Z)
        [0, 0, 0, 1]     # Homogeneous
    ], dtype=torch.float64)


def _standard_to_opengl() -> torch.Tensor:
    """Get transformation matrix from standard to OpenGL coordinate system.
    
    This is the inverse of opengl_to_standard_transform().
    
    Standard convention -> OpenGL convention:
    - Standard X (right) -> OpenGL X (right)
    - Standard Y (forward) -> OpenGL -Z (backward)
    - Standard Z (up) -> OpenGL Y (up)
    
    Returns:
        4x4 transformation matrix
    """
    return torch.tensor([
        [1, 0, 0, 0],    # OpenGL X = Standard X (right stays right)
        [0, 0, 1, 0],    # OpenGL Y = Standard Z (up: Z becomes Y)
        [0, -1, 0, 0],   # OpenGL Z = Standard -Y (backward: +Y becomes -Z)
        [0, 0, 0, 1]     # Homogeneous
    ], dtype=torch.float64)


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
    
    The transformation maps coordinate axes:
    - OpenGL X (right) -> OpenCV X (right)
    - OpenGL Y (up) -> OpenCV -Y (down)
    - OpenGL Z (backward) -> OpenCV -Z (forward)
    
    Returns:
        4x4 transformation matrix
    """
    return torch.tensor([
        [1, 0, 0, 0],    # OpenCV X = OpenGL X (right stays right)
        [0, -1, 0, 0],   # OpenCV Y = OpenGL -Y (up becomes down)
        [0, 0, -1, 0],   # OpenCV Z = OpenGL -Z (backward becomes forward)
        [0, 0, 0, 1]     # Homogeneous
    ], dtype=torch.float64)


def _opencv_to_opengl() -> torch.Tensor:
    """Get transformation matrix from OpenCV to OpenGL coordinate system.
    
    This is the inverse of opengl_to_opencv_transform().
    
    OpenCV convention -> OpenGL convention:
    - OpenCV X (right) -> OpenGL X (right)
    - OpenCV Y (down) -> OpenGL -Y (up)
    - OpenCV Z (forward) -> OpenGL -Z (backward)
    
    Returns:
        4x4 transformation matrix
    """
    return torch.tensor([
        [1, 0, 0, 0],    # OpenGL X = OpenCV X (right stays right)
        [0, -1, 0, 0],   # OpenGL Y = OpenCV -Y (down becomes up)
        [0, 0, -1, 0],   # OpenGL Z = OpenCV -Z (forward becomes backward)
        [0, 0, 0, 1]     # Homogeneous
    ], dtype=torch.float64)


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
    
    The transformation maps coordinate axes:
    - OpenCV X (right) -> Standard X (right)
    - OpenCV Y (down) -> Standard -Z (down)
    - OpenCV Z (forward) -> Standard Y (forward)
    
    This preserves the camera's orientation in world space:
    - OpenCV forward (+Z) -> Standard forward (+Y)
    - OpenCV down (+Y) -> Standard down (-Z)
    - OpenCV right (+X) -> Standard right (+X)
    
    Returns:
        4x4 transformation matrix
    """
    return torch.tensor([
        [1, 0, 0, 0],    # Standard X = OpenCV X (right stays right)
        [0, 0, 1, 0],    # Standard Y = OpenCV Z (forward: Z becomes Y)
        [0, -1, 0, 0],   # Standard Z = OpenCV -Y (up: down becomes up)
        [0, 0, 0, 1]     # Homogeneous
    ], dtype=torch.float64)


def _standard_to_opencv() -> torch.Tensor:
    """Get transformation matrix from standard to OpenCV coordinate system.
    
    This is the inverse of opencv_to_standard_transform().
    
    Standard convention -> OpenCV convention:
    - Standard X (right) -> OpenCV X (right)
    - Standard Y (forward) -> OpenCV Z (forward)
    - Standard Z (up) -> OpenCV -Y (down)
    
    Returns:
        4x4 transformation matrix
    """
    return torch.tensor([
        [1, 0, 0, 0],    # OpenCV X = Standard X (right stays right)
        [0, 0, -1, 0],   # OpenCV Y = Standard -Z (down: up becomes down)
        [0, 1, 0, 0],    # OpenCV Z = Standard Y (forward: Y becomes Z)
        [0, 0, 0, 1]     # Homogeneous
    ], dtype=torch.float64)


def apply_coordinate_transform(
    camera_extrinsics: torch.Tensor, 
    source_convention: str = "opengl",
    target_convention: str = "standard"
) -> torch.Tensor:
    """Transform camera pose between different coordinate system conventions.
    
    Supported coordinate conventions:
    - "standard": X=right, Y=forward, Z=up (camera looks down +Y axis)
    - "opengl": X=right, Y=up, Z=backward (camera looks down -Z axis)  
    - "opencv": X=right, Y=down, Z=forward (camera looks down +Z axis)
    
    Args:
        camera_extrinsics: 4x4 camera extrinsics matrix
        source_convention: Source coordinate convention ("opengl", "standard", "opencv")
        target_convention: Target coordinate convention ("opengl", "standard", "opencv")
        
    Returns:
        Transformed 4x4 camera pose in target convention
        
    Raises:
        ValueError: If unknown coordinate convention is specified
    """
    assert isinstance(camera_extrinsics, torch.Tensor), f"camera_pose must be torch.Tensor, got {type(camera_extrinsics)}"
    assert camera_extrinsics.shape == (4, 4), f"camera_pose must be of shape (4, 4), got {camera_extrinsics.shape}"
    assert camera_extrinsics.is_floating_point(), f"camera_pose must be a floating point tensor, got {camera_extrinsics.dtype}"

    assert source_convention in ["standard", "opengl", "opencv"], f"Unknown source_convention: {source_convention}"
    assert target_convention in ["standard", "opengl", "opencv"], f"Unknown target_convention: {target_convention}"

    if source_convention == target_convention:
        return camera_extrinsics
    
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
    else:
        raise ValueError(
            f"Unsupported transformation: {source_convention} -> {target_convention}. "
            f"Supported: 'opengl' <-> 'standard', 'opengl' <-> 'opencv', 'opencv' <-> 'standard'"
        )
    
    # Move transform to same device and dtype as camera pose
    if camera_extrinsics.device != transform.device or camera_extrinsics.dtype != transform.dtype:
        transform = transform.to(device=camera_extrinsics.device, dtype=camera_extrinsics.dtype)
    
    camera_extrinsics_transformed = camera_extrinsics @ torch.inverse(materialize_tensor(transform))
    
    return camera_extrinsics_transformed
