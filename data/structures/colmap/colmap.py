from pathlib import Path
from typing import Any, Dict

import numpy as np

from data.structures.colmap.load import load_colmap_data
from data.structures.colmap.save import save_colmap_data
from data.structures.colmap.transform import (
    transform_colmap_cameras,
    transform_colmap_points,
)


class COLMAP_Data:

    def __init__(
        self,
        model_dir: str | Path,
    ) -> None:
        # Input validations
        assert isinstance(model_dir, (str, Path)), f"{type(model_dir)=}"

        model_path = Path(model_dir)
        assert model_path.is_dir(), f"COLMAP model dir not found: {model_path}"

        cameras_bin = model_path / "cameras.bin"
        images_bin = model_path / "images.bin"
        points_bin = model_path / "points3D.bin"
        cameras_txt = model_path / "cameras.txt"
        images_txt = model_path / "images.txt"
        points_txt = model_path / "points3D.txt"

        has_binary = (
            cameras_bin.exists() and images_bin.exists() and points_bin.exists()
        )
        has_text = cameras_txt.exists() and images_txt.exists() and points_txt.exists()
        assert has_binary and has_text, (
            "COLMAP model must include both binary and text files: " f"{model_path}"
        )

        cameras, images, points3D = load_colmap_data(model_dir=model_path)

        self._validate_cameras(cameras)
        self._validate_images(images)
        self._validate_points(points3D)
        self._validate_image_camera_links(images=images, cameras=cameras)

        self.model_dir = model_path
        self.cameras = cameras
        self.images = images
        self.points3D = points3D

    @classmethod
    def from_data(
        cls,
        model_dir: str | Path,
        cameras: Dict[int, Any],
        images: Dict[int, Any],
        points3D: Dict[int, Any],
    ) -> Any:
        # Input validations
        assert isinstance(model_dir, (str, Path)), f"{type(model_dir)=}"
        assert isinstance(cameras, dict), f"{type(cameras)=}"
        assert isinstance(images, dict), f"{type(images)=}"
        assert isinstance(points3D, dict), f"{type(points3D)=}"

        model_path = Path(model_dir)
        colmap_data = cls.__new__(cls)
        colmap_data.model_dir = model_path
        colmap_data.cameras = cameras
        colmap_data.images = images
        colmap_data.points3D = points3D

        colmap_data._validate_cameras(cameras)
        colmap_data._validate_images(images)
        colmap_data._validate_points(points3D)
        colmap_data._validate_image_camera_links(images=images, cameras=cameras)

        return colmap_data

    def transform(
        self,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> None:
        # Input validations
        assert isinstance(scale, (int, float)), f"{type(scale)=}"
        assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
        assert rotation.shape == (3, 3), f"{rotation.shape=}"
        assert rotation.dtype == np.float32, f"{rotation.dtype=}"
        assert isinstance(translation, np.ndarray), f"{type(translation)=}"
        assert translation.shape == (3,), f"{translation.shape=}"
        assert translation.dtype == np.float32, f"{translation.dtype=}"

        self.images = transform_colmap_cameras(
            images=self.images,
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        self.points3D = transform_colmap_points(
            points=self.points3D,
            scale=scale,
            rotation=rotation,
            translation=translation,
        )

    def save(self) -> None:
        save_colmap_data(
            output_dir=self.model_dir,
            cameras=self.cameras,
            images=self.images,
            points3D=self.points3D,
        )

    @staticmethod
    def _validate_cameras(cameras: Dict[int, Any]) -> None:
        # Input validations
        assert isinstance(cameras, dict), f"{type(cameras)=}"

        for camera_id, camera in cameras.items():
            assert isinstance(camera_id, int), f"{type(camera_id)=}"
            assert camera.id == camera_id, f"{camera_id=} {camera.id=}"

    @staticmethod
    def _validate_images(images: Dict[int, Any]) -> None:
        # Input validations
        assert isinstance(images, dict), f"{type(images)=}"

        for image_id, image in images.items():
            assert isinstance(image_id, int), f"{type(image_id)=}"
            assert image.id == image_id, f"{image_id=} {image.id=}"

    @staticmethod
    def _validate_points(points3D: Dict[int, Any]) -> None:
        # Input validations
        assert isinstance(points3D, dict), f"{type(points3D)=}"

        for point_id, point in points3D.items():
            assert isinstance(point_id, int), f"{type(point_id)=}"
            assert point.id == point_id, f"{point_id=} {point.id=}"

    @staticmethod
    def _validate_image_camera_links(
        images: Dict[int, Any],
        cameras: Dict[int, Any],
    ) -> None:
        # Input validations
        assert isinstance(images, dict), f"{type(images)=}"
        assert isinstance(cameras, dict), f"{type(cameras)=}"

        for image in images.values():
            assert image.camera_id in cameras, f"{image.camera_id=}"
