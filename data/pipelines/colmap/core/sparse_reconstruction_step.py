"""Step that runs COLMAP sparse reconstruction."""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from data.pipelines.base_step import BaseStep
from data.structures.colmap.load import (
    _load_colmap_cameras_bin,
    _load_colmap_images_bin,
    _load_colmap_points_bin,
)


class ColmapSparseReconstructionStep(BaseStep):
    """Copy from NerfStudio pipeline: COLMAP mapper stage."""

    STEP_NAME = "colmap_sparse_reconstruction"

    def __init__(self, scene_root: str | Path) -> None:
        scene_root = Path(scene_root)
        self.input_images_dir = scene_root / "input"
        self.distorted_dir = scene_root / "distorted"
        self.sparse_output_dir = scene_root / "distorted" / "sparse"
        super().__init__(input_root=scene_root, output_root=scene_root)

    def _init_output_files(self) -> None:
        self.output_files = [
            "distorted/sparse/0/cameras.bin",
            "distorted/sparse/0/images.bin",
            "distorted/sparse/0/points3D.bin",
        ]

    def _init_input_files(self) -> None:
        image_names = self._input_image_names()
        self.input_files = [f"input/{name}" for name in image_names]
        self.input_files.append("distorted/database.db")

    def check_outputs(self) -> bool:
        outputs_ready = super().check_outputs()
        if not outputs_ready:
            return False
        try:
            self._validate_sparse_files()
        except Exception as e:
            logging.debug("Sparse reconstruction validation failed: %s", e)
            return False
        else:
            return True

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.sparse_output_dir.mkdir(parents=True, exist_ok=True)
        if not force and self.check_outputs():
            logging.info("🏗️ COLMAP sparse reconstruction already done - SKIPPED")
            return {}
        logging.info("   🏗️ Sparse reconstruction")
        distorted_db_path = self.distorted_dir / "database.db"
        mapper_cmd = (
            f"colmap mapper "
            f"--database_path {distorted_db_path} "
            f"--image_path {self.input_images_dir} "
            f"--output_path {self.sparse_output_dir} "
            f"--Mapper.multiple_models 0 "
            f"--Mapper.ba_global_function_tolerance=0.000001 "
            f"--Mapper.tri_ignore_two_view_tracks 0 "
            f"--Mapper.tri_min_angle 1 "
            "--log_to_stderr 1"
        )
        ret_code = subprocess.call(mapper_cmd, shell=True)
        assert ret_code == 0, f"COLMAP mapper failed with code {ret_code}"
        self._validate_sparse_files()
        return {}

    def _input_image_names(self) -> List[str]:
        entries = sorted(self.input_images_dir.iterdir())
        assert entries, f"Empty input dir or no files: {self.input_images_dir}"
        assert all(entry.is_file() for entry in entries), (
            "COLMAP input directory must only contain files "
            f"(found non-file entries in {self.input_images_dir})"
        )
        return [entry.name for entry in entries]

    def _validate_sparse_files(self) -> None:
        cameras_path = self.sparse_output_dir / "0" / "cameras.bin"
        images_path = self.sparse_output_dir / "0" / "images.bin"
        points_path = self.sparse_output_dir / "0" / "points3D.bin"
        assert cameras_path.exists(), f"cameras.bin not found: {cameras_path}"
        assert images_path.exists(), f"images.bin not found: {images_path}"
        assert points_path.exists(), f"points3D.bin not found: {points_path}"

        cameras = _load_colmap_cameras_bin(path_to_model_file=str(cameras_path))
        images = _load_colmap_images_bin(path_to_model_file=str(images_path))
        points3d = _load_colmap_points_bin(path_to_model_file=str(points_path))

        assert cameras, f"No cameras parsed from {cameras_path}"
        assert (
            len(cameras) == 1
        ), f"Expected exactly one camera in {cameras_path}, found {len(cameras)}"
        assert images, f"No registered images parsed from {images_path}"
        expected_names = set(self._input_image_names())
        registered_names = {img.name for img in images.values()}
        assert registered_names == expected_names, (
            f"Registered image names in images.bin do not match input images. "
            f"expected={len(expected_names)} actual={len(registered_names)}"
        )
        assert points3d, f"No points parsed from {points_path}"
        image_ids = {img.id for img in images.values()}
        # for point in points3d.values():
        #     assert len(point.image_ids) == len(
        #         point.point2D_idxs
        #     ), "points3D.bin contains mismatched image_ids and point2D_idxs lengths"
        #     assert set(point.image_ids).issubset(
        #         image_ids
        #     ), "points3D.bin references image ids not present in images.bin"
