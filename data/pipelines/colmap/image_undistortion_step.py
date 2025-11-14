"""Step that runs COLMAP image undistortion."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from project.pipelines.base_step import BaseStep


class ColmapImageUndistortionStep(BaseStep):
    """Copy from NerfStudio pipeline: COLMAP image_undistorter stage."""

    STEP_NAME = "colmap_image_undistortion"

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        super().__init__(input_root=input_root, output_root=output_root)
        self.images_dir = self.output_root / "images"
        self.temp_sparse_dir = self.output_root / "sparse"
        self.distorted_sparse_dir = self.output_root / "distorted" / "sparse"
        self.undistorted_sparse_dir = self.output_root / "undistorted" / "sparse"

    def check_outputs(self) -> bool:
        images_exist = self.images_dir.is_dir() and any(self.images_dir.iterdir())
        temp_ok = (self.temp_sparse_dir / "0" / "cameras.bin").exists()
        undistorted_ok = (self.undistorted_sparse_dir / "0" / "cameras.bin").exists()
        return images_exist and (temp_ok or undistorted_ok)

    def run(self, force: bool = False) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.temp_sparse_dir.mkdir(parents=True, exist_ok=True)
        if not force and self.check_outputs():
            logging.info("📐 COLMAP undistortion already done - SKIPPED")
            return
        logging.info("   📐 Image undistortion")
        undistort_cmd = (
            f"colmap image_undistorter "
            f"--image_path {self.input_root} "
            f"--input_path {self.distorted_sparse_dir / '0'} "
            f"--output_path {self.output_root} "
            f"--output_type COLMAP"
        )
        ret_code = subprocess.call(undistort_cmd, shell=True)
        assert ret_code == 0, f"COLMAP image undistortion failed with code {ret_code}"
