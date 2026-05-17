"""TypeScript backend camera-state conversion."""

from data.structures.three_d.camera import Camera
from data.viewer.utils.camera_state.ts.backend.schemas.camera_state import CameraState


def create_camera_state_from_camera(camera: Camera) -> CameraState:
    """Create a frontend camera-state schema from a Camera.

    Args:
        camera: Camera object whose tensors use the camera's declared convention.

    Returns:
        Serialized camera state preserving intrinsics, extrinsics, convention, name, and id.
    """
    assert isinstance(camera, Camera), "Camera must be a Camera. camera=%r" % camera
    return CameraState(
        intrinsics={"matrix": camera.intrinsics.detach().cpu().tolist()},
        extrinsics={"matrix": camera.extrinsics.detach().cpu().tolist()},
        convention=camera.convention,
        name=camera.name,
        id=None if camera.id is None else str(camera.id),
    )
