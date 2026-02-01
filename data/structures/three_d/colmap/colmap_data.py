from pathlib import Path
from typing import Any, Dict

import numpy as np

from data.structures.three_d.colmap.load import load_colmap_data
from data.structures.three_d.colmap.save import save_colmap_data
from data.structures.three_d.colmap.transform import transform_colmap
from data.structures.three_d.colmap.validate import (
    validate_cameras,
    validate_image_camera_links,
    validate_images,
    validate_points3D,
)


class COLMAP_Data:

    def __init__(
        self,
        cameras: Dict[int, Any],
        images: Dict[int, Any],
        points3D: Dict[int, Any],
    ) -> None:
        validate_cameras(cameras)
        validate_images(images)
        validate_points3D(points3D)
        validate_image_camera_links(images=images, cameras=cameras)

        self.cameras = cameras
        self.images = images
        self.points3D = points3D

    @classmethod
    def load(cls, model_dir: str | Path) -> "COLMAP_Data":
        # Input validations
        assert isinstance(model_dir, (str, Path)), f"{type(model_dir)=}"

        # Input normalizations
        model_path = Path(model_dir)

        cameras, images, points3D = load_colmap_data(model_dir=model_path)
        return cls(cameras=cameras, images=images, points3D=points3D)

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

        cameras, images, points3D = transform_colmap(
            cameras=self.cameras,
            images=self.images,
            points=self.points3D,
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        self.cameras = cameras
        self.images = images
        self.points3D = points3D

    def save(self, output_dir: str | Path) -> None:
        # Input validations
        assert isinstance(output_dir, (str, Path)), f"{type(output_dir)=}"

        # Input normalizations
        output_path = Path(output_dir)

        save_colmap_data(data=self, output_path=output_path)
