"""
DATA.STRUCTURES.THREE_D.CAMERA.INTRINSICS API
"""

from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    CameraIntrinsics,
    CameraIntrinsicsOrtho,
    CameraIntrinsicsPinhole,
    CameraIntrinsicsSimplePinhole,
    build_camera_intrinsics,
)
from data.structures.three_d.camera.intrinsics.conventions import (
    transform_intr_convention,
)
from data.structures.three_d.camera.intrinsics.scaling import (
    rescale_intr_params,
    resolve_target_resolution,
)
from data.structures.three_d.camera.intrinsics.validation import (
    validate_camera_intrinsics_attributes,
    validate_camera_intrinsics_invariants,
    validate_camera_intrinsics_params,
    validate_camera_model,
    validate_intr_convention,
)

__all__ = (
    "CameraIntrinsics",
    "CameraIntrinsicsOrtho",
    "CameraIntrinsicsPinhole",
    "CameraIntrinsicsSimplePinhole",
    "build_camera_intrinsics",
    "transform_intr_convention",
    "rescale_intr_params",
    "resolve_target_resolution",
    "validate_camera_intrinsics_attributes",
    "validate_camera_intrinsics_invariants",
    "validate_camera_intrinsics_params",
    "validate_camera_model",
    "validate_intr_convention",
)
