"""Coordinate system transformation utilities for point cloud rendering.

This module provides utilities to handle different camera coordinate system conventions
and transform them to standardized coordinate systems used in point cloud rendering.
"""

import torch
from typing import Union


def opengl_to_standard_transform() -> torch.Tensor:
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
    ], dtype=torch.float32)


def standard_to_opengl_transform() -> torch.Tensor:
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
    ], dtype=torch.float32)


def apply_coordinate_transform(
    camera_pose: torch.Tensor, 
    source_convention: str = "opengl",
    target_convention: str = "standard"
) -> torch.Tensor:
    """Transform camera pose between different coordinate system conventions.
    
    Args:
        camera_pose: 4x4 camera extrinsics matrix
        source_convention: Source coordinate convention ("opengl", "standard")
        target_convention: Target coordinate convention ("opengl", "standard")
        
    Returns:
        Transformed 4x4 camera pose in target convention
        
    Raises:
        ValueError: If unknown coordinate convention is specified
    """
    if source_convention == target_convention:
        return camera_pose
    
    # Get appropriate transformation matrix
    if source_convention == "opengl" and target_convention == "standard":
        transform = opengl_to_standard_transform()
    elif source_convention == "standard" and target_convention == "opengl":
        transform = standard_to_opengl_transform()
    else:
        raise ValueError(
            f"Unsupported transformation: {source_convention} -> {target_convention}. "
            f"Supported: 'opengl' <-> 'standard'"
        )
    
    # Move transform to same device as camera pose
    if camera_pose.device != transform.device:
        transform = transform.to(camera_pose.device)
    
    # Apply coordinate transformation to the entire pose matrix
    # This transforms both the rotation and translation components properly
    camera_pose_transformed = transform @ camera_pose
    
    return camera_pose_transformed


def transform_camera_pose(
    camera_pose: Union[torch.Tensor, "np.ndarray"],
    convention: str = "opengl"
) -> torch.Tensor:
    """Legacy function for backward compatibility. Use apply_coordinate_transform instead.
    
    Args:
        camera_pose: 4x4 camera extrinsics matrix
        convention: Source coordinate convention ("opengl", "standard")
        
    Returns:
        Transformed 4x4 camera pose in standard convention
    """
    if hasattr(camera_pose, 'numpy'):  # torch.Tensor
        pose_tensor = camera_pose
    else:  # numpy array
        import torch
        pose_tensor = torch.from_numpy(camera_pose).float()
    
    return apply_coordinate_transform(
        pose_tensor, 
        source_convention=convention, 
        target_convention="standard"
    )