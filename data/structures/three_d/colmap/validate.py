from typing import Any, Dict


def validate_cameras(cameras: Dict[int, Any]) -> None:
    # Input validations
    assert isinstance(cameras, dict), f"{type(cameras)=}"

    for camera_id, camera in cameras.items():
        assert isinstance(camera_id, int), f"{type(camera_id)=}"
        assert camera.id == camera_id, f"{camera_id=} {camera.id=}"


def validate_images(images: Dict[int, Any]) -> None:
    # Input validations
    assert isinstance(images, dict), f"{type(images)=}"

    for image_id, image in images.items():
        assert isinstance(image_id, int), f"{type(image_id)=}"
        assert image.id == image_id, f"{image_id=} {image.id=}"


def validate_points3D(points3D: Dict[int, Any]) -> None:
    # Input validations
    assert isinstance(points3D, dict), f"{type(points3D)=}"

    for point_id, point in points3D.items():
        assert isinstance(point_id, int), f"{type(point_id)=}"
        assert point.id == point_id, f"{point_id=} {point.id=}"


def validate_image_camera_links(
    images: Dict[int, Any],
    cameras: Dict[int, Any],
) -> None:
    # Input validations
    assert isinstance(images, dict), f"{type(images)=}"
    assert isinstance(cameras, dict), f"{type(cameras)=}"

    for image in images.values():
        assert image.camera_id in cameras, f"{image.camera_id=}"
