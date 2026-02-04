"""Step that runs COLMAP point_triangulator using DJI-initialized poses."""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from data.pipelines.base_step import BaseStep
from data.structures.three_d.colmap.load import (
    _load_colmap_cameras_bin,
    _load_colmap_images_bin,
    _load_colmap_points_bin,
)


class ColmapPointTriangulationStep(BaseStep):
    """Triangulate points starting from an initialized sparse model."""

    STEP_NAME = "colmap_point_triangulation"

    def __init__(self, scene_root: str | Path) -> None:
        scene_root = Path(scene_root)
        self.input_images_dir = scene_root / "input"
        self.database_path = scene_root / "distorted" / "database.db"
        self.init_model_dir = scene_root / "distorted" / "init_model"
        self.output_model_dir = scene_root / "distorted" / "sparse" / "0"
        super().__init__(input_root=scene_root, output_root=scene_root)

    def _init_input_files(self) -> None:
        entries = sorted(self.input_images_dir.iterdir())
        assert entries, f"Empty input dir or no files: {self.input_images_dir}"
        assert all(entry.is_file() for entry in entries), (
            "COLMAP input directory must only contain files "
            f"(found non-file entries in {self.input_images_dir})"
        )
        self.image_names = [entry.name for entry in entries]
        filenames: List[str] = [f"input/{name}" for name in self.image_names]
        filenames.append("distorted/database.db")
        filenames.extend(
            [
                "distorted/init_model/cameras.bin",
                "distorted/init_model/images.bin",
                "distorted/init_model/points3D.bin",
            ]
        )
        self.input_files = filenames

    def _init_output_files(self) -> None:
        self.output_files = [
            "distorted/sparse/0/cameras.bin",
            "distorted/sparse/0/images.bin",
            "distorted/sparse/0/points3D.bin",
        ]

    def build(self, force: bool = False) -> None:
        super().build(force=force)
        self.run(kwargs={}, force=force)

    def check_outputs(self) -> bool:
        outputs_ready = super().check_outputs()
        if not outputs_ready:
            return False
        try:
            self._validate_outputs()
            return True
        except Exception as e:
            logging.debug("Point triangulation validation failed: %s", e)
            return False

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        if not force and self.check_outputs():
            return {}

        self.output_model_dir.mkdir(parents=True, exist_ok=True)

        cmd_parts = self._build_colmap_command()
        result = subprocess.run(cmd_parts, capture_output=True, text=True)
        ret_code = result.returncode
        assert ret_code == 0, (
            f"COLMAP point_triangulator failed with code {ret_code}. "
            f"STDOUT: {result.stdout} STDERR: {result.stderr}"
        )

        self._validate_outputs()
        return {}

    def _build_colmap_command(self) -> List[str]:
        return [
            "colmap",
            "point_triangulator",
            "--database_path",
            str(self.database_path),
            "--image_path",
            str(self.input_images_dir),
            "--input_path",
            str(self.init_model_dir),
            "--output_path",
            str(self.output_model_dir),
            "--clear_points",
            "1",
            "--refine_intrinsics",
            "0",
            "--log_to_stderr",
            "1",
        ]

    def _validate_outputs(self) -> None:
        self._validate_colmap_cameras()
        self._validate_colmap_images()
        self._validate_colmap_points()

    def _validate_colmap_cameras(self) -> None:
        cameras_path = self.output_model_dir / "cameras.bin"
        cameras = _load_colmap_cameras_bin(path_to_model_file=str(cameras_path))
        assert cameras, f"No cameras parsed from {cameras_path}"
        assert (
            len(cameras) == 1
        ), f"Expected exactly one camera in {cameras_path}, found {len(cameras)}"

    def _validate_colmap_images(self) -> None:
        images_path = self.output_model_dir / "images.bin"
        images = _load_colmap_images_bin(path_to_model_file=str(images_path))
        assert images, f"No registered images parsed from {images_path}"
        registered_names = sorted(image.name for image in images.values())
        assert registered_names == sorted(self.image_names), (
            "Triangulated model image names do not match input PNGs. "
            f"expected={len(self.image_names)} actual={len(registered_names)}"
        )

    def _validate_colmap_points(self) -> None:
        points_path = self.output_model_dir / "points3D.bin"
        points3d = _load_colmap_points_bin(path_to_model_file=str(points_path))
        assert points3d, f"No points parsed from {points_path}"
