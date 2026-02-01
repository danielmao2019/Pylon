"""
DATA.STRUCTURES.THREE_D.CAMERA.ROTATION API
"""

from data.structures.three_d.camera.rotation.euler import (
    euler_canonical,
    euler_to_matrix,
    matrix_to_euler,
)
from data.structures.three_d.camera.rotation.pitch_yaw import (
    matrix_to_pitch_yaw,
    pitch_yaw_to_matrix,
)
from data.structures.three_d.camera.rotation.quaternion import (
    quat_to_rotmat,
    qvec2rotmat,
    rotmat2qvec,
    rotmat_to_quat,
)
from data.structures.three_d.camera.rotation.rodrigues import (
    matrix_to_rodrigues,
    rodrigues_canonical,
    rodrigues_to_matrix,
)
from data.structures.three_d.camera.rotation.zero_roll import (
    _GLOBAL_UP,
    _find_local_up,
    zero_roll,
)

__all__ = (
    "euler_canonical",
    "euler_to_matrix",
    "matrix_to_euler",
    "matrix_to_pitch_yaw",
    "pitch_yaw_to_matrix",
    "quat_to_rotmat",
    "qvec2rotmat",
    "rotmat2qvec",
    "rotmat_to_quat",
    "matrix_to_rodrigues",
    "rodrigues_canonical",
    "rodrigues_to_matrix",
    "_GLOBAL_UP",
    "_find_local_up",
    "zero_roll",
)
