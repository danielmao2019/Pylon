import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from data.pipelines.base_step import BaseStep
from data.pipelines.colmap.coords.umeyama import compute_umeyama_alignment
from data.pipelines.colmap.coords.validate import (
    validate_camera_center_alignment,
)
from data.structures.three_d.colmap import COLMAP_Data
from data.structures.three_d.nerfstudio import NerfStudio_Data
from data.structures.three_d.point_cloud import load_point_cloud, save_point_cloud
from data.structures.three_d.point_cloud.ops.apply_transform import apply_transform


class BaseNormalizeCoordsStep(BaseStep, ABC):

    def check_outputs(self) -> bool:
        if not super().check_outputs():
            return False
        try:
            self._validate_sparse_model_dir(
                sparse_root=self.output_root / "undistorted" / "sparse"
            )
            self._validate_coords()
            return True
        except Exception as e:
            logging.debug("Coords normalization validation failed: %s", e)
            return False

    def _validate_sparse_model_dir(self, sparse_root: Path) -> None:
        # Input validations
        assert isinstance(sparse_root, Path), f"{type(sparse_root)=}"

        cameras_path = sparse_root / "cameras.bin"
        images_path = sparse_root / "images.bin"
        points_path = sparse_root / "points3D.bin"
        assert cameras_path.exists(), f"cameras.bin not found: {cameras_path}"
        assert images_path.exists(), f"images.bin not found: {images_path}"
        assert points_path.exists(), f"points3D.bin not found: {points_path}"

    def _validate_coords(self) -> None:
        source_camera_names, source_camera_centers = self._get_source_camera_centers()
        target_camera_names, target_camera_centers = self._get_target_camera_centers()
        validate_camera_center_alignment(
            source_camera_names=source_camera_names,
            source_camera_centers=source_camera_centers,
            target_camera_names=target_camera_names,
            target_camera_centers=target_camera_centers,
        )

    def _compute_transform(self) -> Tuple[float, np.ndarray, np.ndarray]:
        source_camera_names, source_camera_centers = self._get_source_camera_centers()
        target_camera_names, target_camera_centers = self._get_target_camera_centers()
        assert (
            source_camera_names == target_camera_names
        ), "Source/target camera-name order mismatch for Umeyama alignment"
        assert (
            source_camera_centers.dtype == np.float32
        ), f"{source_camera_centers.dtype=}"
        assert (
            target_camera_centers.dtype == np.float32
        ), f"{target_camera_centers.dtype=}"
        scale, rotation, translation = compute_umeyama_alignment(
            source_points=source_camera_centers.T,
            target_points=target_camera_centers.T,
        )
        assert isinstance(scale, float), f"{type(scale)=}"
        assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
        assert isinstance(translation, np.ndarray), f"{type(translation)=}"
        assert rotation.dtype == np.float32, f"{rotation.dtype=}"
        assert translation.dtype == np.float32, f"{translation.dtype=}"
        return scale, rotation, translation

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        if not force and self.check_outputs():
            return {}
        scale, rotation, translation = self._compute_transform()
        self._transform_colmap(
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        self._transform_nerfstudio(
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        return {}

    def _transform_colmap(
        self,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> None:
        sparse_root = self.output_root / "undistorted" / "sparse"
        colmap_data = COLMAP_Data.load(model_dir=sparse_root)
        colmap_data.transform(
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        colmap_data.save(output_dir=sparse_root)

    def _transform_nerfstudio(
        self,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> None:
        transforms_path = self.output_root / "transforms.json"
        transforms = NerfStudio_Data.load(filepath=transforms_path, device="cpu")
        transforms.transform(
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        transforms.save(output_path=transforms_path)
        logging.info("   ✓ Updated transforms.json with aligned poses")
        self._transform_nerfstudio_point_cloud(
            scale=scale,
            rotation=rotation,
            translation=translation,
        )

    def _transform_nerfstudio_point_cloud(
        self,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> None:
        raw_ply_path = (self.output_root / "sparse_pc.ply").resolve()
        assert raw_ply_path.is_file(), f"Missing sparse point cloud: {raw_ply_path}"

        point_cloud = load_point_cloud(str(raw_ply_path), device="cuda")
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = scale * rotation
        transform[:3, 3] = translation
        point_cloud.xyz = apply_transform(point_cloud.xyz, transform)
        save_point_cloud(point_cloud, str(raw_ply_path))
        logging.info(
            "   ✓ Applied alignment transform to sparse point cloud: %s", raw_ply_path
        )

    @abstractmethod
    def _get_source_camera_centers(
        self,
    ) -> Tuple[List[str], np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def _get_target_camera_centers(
        self,
    ) -> Tuple[List[str], np.ndarray]:
        raise NotImplementedError
