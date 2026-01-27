from pathlib import Path
from typing import Any, Dict

import numpy as np

from data.structures.colmap.load import (
    load_colmap_model,
    save_colmap_model_binary,
    save_colmap_model_text,
)
from utils.three_d.rotation.quaternion import qvec2rotmat, rotmat2qvec


class COLMAP_Data:

    def __init__(
        self,
        model_dir: str | Path,
        file_format: str = "binary",
    ) -> None:
        # Input validations
        assert isinstance(model_dir, (str, Path)), f"{type(model_dir)=}"
        assert isinstance(file_format, str), f"{type(file_format)=}"
        assert file_format in ("binary", "text"), f"{file_format=}"

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
        assert has_binary or has_text, f"No COLMAP model files found in {model_path}"

        if file_format == "binary":
            assert has_binary, f"COLMAP binary files not found in {model_path}"
        else:
            assert has_text, f"COLMAP text files not found in {model_path}"
        cameras, images, points3D = load_colmap_model(
            model_dir=model_path,
            file_format=file_format,
        )

        self._validate_cameras(cameras)
        self._validate_images(images)
        self._validate_points(points3D)
        self._validate_image_camera_links(images=images, cameras=cameras)

        self.model_dir = model_path
        self.file_format = file_format
        self._has_binary = has_binary
        self._has_text = has_text
        self.cameras = cameras
        self.images = images
        self.points3D = points3D

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

        self.images = self._transform_cameras(
            images=self.images,
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        self.points3D = self._transform_points(
            points=self.points3D,
            scale=scale,
            rotation=rotation,
            translation=translation,
        )

    def save(self) -> None:
        self._save()

    def _save(self) -> None:
        if self._has_binary:
            save_colmap_model_binary(
                output_dir=self.model_dir,
                cameras=self.cameras,
                images=self.images,
                points3D=self.points3D,
            )
        if self._has_text:
            save_colmap_model_text(
                output_dir=self.model_dir,
                cameras=self.cameras,
                images=self.images,
                points3D=self.points3D,
            )

    @staticmethod
    def _transform_cameras(
        images: Dict[int, Any],
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> Dict[int, Any]:
        # Input validations
        assert isinstance(images, dict), f"{type(images)=}"
        assert isinstance(scale, (int, float)), f"{type(scale)=}"
        assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
        assert rotation.shape == (3, 3), f"{rotation.shape=}"
        assert rotation.dtype == np.float32, f"{rotation.dtype=}"
        assert isinstance(translation, np.ndarray), f"{type(translation)=}"
        assert translation.shape == (3,), f"{translation.shape=}"
        assert translation.dtype == np.float32, f"{translation.dtype=}"

        aligned_images: Dict[int, Any] = {}
        for image_id, image in images.items():
            assert isinstance(image_id, int), f"{type(image_id)=}"
            assert image.id == image_id, f"{image_id=} {image.id=}"
            assert isinstance(image.qvec, np.ndarray), f"{type(image.qvec)=}"
            assert image.qvec.shape == (4,), f"{image.qvec.shape=}"
            assert isinstance(image.tvec, np.ndarray), f"{type(image.tvec)=}"
            assert image.tvec.shape == (3,), f"{image.tvec.shape=}"

            R_w2c = qvec2rotmat(image.qvec)
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ image.tvec
            R_c2w_new = rotation @ R_c2w
            t_c2w_new = scale * (rotation @ t_c2w) + translation
            R_w2c_new = R_c2w_new.T
            t_w2c_new = -R_w2c_new @ t_c2w_new
            qvec_new = rotmat2qvec(R_w2c_new)

            aligned_images[image_id] = type(image)(
                id=image.id,
                qvec=qvec_new,
                tvec=t_w2c_new,
                camera_id=image.camera_id,
                name=image.name,
                xys=image.xys,
                point3D_ids=image.point3D_ids,
            )
        return aligned_images

    @staticmethod
    def _transform_points(
        points: Dict[int, Any],
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> Dict[int, Any]:
        # Input validations
        assert isinstance(points, dict), f"{type(points)=}"
        assert isinstance(scale, (int, float)), f"{type(scale)=}"
        assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
        assert rotation.shape == (3, 3), f"{rotation.shape=}"
        assert rotation.dtype == np.float32, f"{rotation.dtype=}"
        assert isinstance(translation, np.ndarray), f"{type(translation)=}"
        assert translation.shape == (3,), f"{translation.shape=}"
        assert translation.dtype == np.float32, f"{translation.dtype=}"

        aligned_points: Dict[int, Any] = {}
        for point_id, point in points.items():
            assert isinstance(point_id, int), f"{type(point_id)=}"
            assert point.id == point_id, f"{point_id=} {point.id=}"
            assert isinstance(point.xyz, np.ndarray), f"{type(point.xyz)=}"
            assert point.xyz.shape == (3,), f"{point.xyz.shape=}"

            xyz_new = scale * (rotation @ point.xyz) + translation
            aligned_points[point_id] = type(point)(
                id=point.id,
                xyz=xyz_new,
                rgb=point.rgb,
                error=point.error,
                image_ids=point.image_ids,
                point2D_idxs=point.point2D_idxs,
            )
        return aligned_points

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
