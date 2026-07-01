"""
DATA.STRUCTURES.THREE_D.CAMERA.EXTRINSICS API
"""

from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.extrinsics.conventions import transform_convention
from data.structures.three_d.camera.extrinsics.validation import (
    validate_camera_convention,
    validate_camera_extrinsics,
    validate_camera_extrinsics_attributes,
    validate_rotation_matrix,
)

__all__ = (
    "CameraExtrinsics",
    "transform_convention",
    "validate_camera_convention",
    "validate_camera_extrinsics",
    "validate_camera_extrinsics_attributes",
    "validate_rotation_matrix",
)
