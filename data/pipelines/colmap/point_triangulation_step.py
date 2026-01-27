"""Step that runs COLMAP point_triangulator using DJI-initialized poses."""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from data.pipelines.base_step import BaseStep
from data.structures.colmap.load import load_cameras_binary, load_images_binary


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

    def check_outputs(self) -> bool:
        outputs_ready = super().check_outputs()
        if not outputs_ready:
            return False
        try:
            self._validate_registered_images()
            return True
        except Exception as e:
            logging.debug("Point triangulation validation failed: %s", e)
            return False

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.output_model_dir.mkdir(parents=True, exist_ok=True)
        if not force and self.check_outputs():
            logging.info("   🛰️ COLMAP point triangulation already done - SKIPPED")
            return {}
        cmd = (
            f"colmap point_triangulator "
            f"--database_path {self.database_path} "
            f"--image_path {self.input_images_dir} "
            f"--input_path {self.init_model_dir} "
            f"--output_path {self.output_model_dir} "
            f"--clear_points 1 "
            f"--refine_intrinsics 0 "
            f"--log_to_stderr 1"
        )
        ret_code = subprocess.call(cmd, shell=True)
        assert ret_code == 0, f"COLMAP point_triangulator failed with code {ret_code}"
        self._validate_registered_images()
        return {}

    def _validate_registered_images(self) -> None:
        cameras_path = self.output_model_dir / "cameras.bin"
        images_path = self.output_model_dir / "images.bin"
        cameras = load_cameras_binary(str(cameras_path))
        images = load_images_binary(str(images_path))
        assert cameras, f"No cameras parsed from {cameras_path}"
        assert (
            len(cameras) == 1
        ), f"Expected exactly one camera in {cameras_path}, found {len(cameras)}"
        assert images, f"No registered images parsed from {images_path}"
        registered_names = sorted(image.name for image in images.values())
        assert registered_names == sorted(self.image_names), (
            "Triangulated model image names do not match input PNGs. "
            f"expected={len(self.image_names)} actual={len(registered_names)}"
        )
