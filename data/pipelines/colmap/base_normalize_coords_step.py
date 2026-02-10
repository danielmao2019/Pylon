import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from data.pipelines.base_step import BaseStep
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
        scale, rotation, translation = self._compute_transform()
        identity = np.eye(3, dtype=np.float32)
        rotation_diff = float(np.max(np.abs(rotation - identity)))
        scale_diff = abs(scale - 1.0)
        bounds = self._get_alignment_bounds()
        assert set(bounds.keys()) == {
            "min",
            "max",
        }, f"Alignment bounds must contain min/max, got keys {sorted(bounds.keys())}"
        diagonal = float(np.linalg.norm(bounds["max"] - bounds["min"]))
        assert diagonal > 0, "Alignment bounds diagonal must be positive"
        translation_norm = float(np.linalg.norm(translation))
        assert (
            rotation_diff <= 1.0e-03
        ), f"Rotation too far from identity: {rotation_diff}"
        assert scale_diff <= 1.0e-03, f"Scale too far from 1.0: {scale_diff}"
        assert (
            translation_norm <= diagonal * 1.0e-03
        ), f"Translation too large: {translation_norm}"

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
    def _compute_transform(self) -> Tuple[float, np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def _get_alignment_bounds(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def _bounds(points: np.ndarray) -> Dict[str, np.ndarray]:
        # Input validations
        assert isinstance(points, np.ndarray), f"{type(points)=}"
        assert (
            points.ndim == 2 and points.shape[1] == 3
        ), f"Expected (N,3) points array, got shape {points.shape}"
        assert points.size > 0, "No points provided for bounds computation"

        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        return {"min": mins, "max": maxs}
