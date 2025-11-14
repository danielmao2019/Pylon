"""Step that runs COLMAP sparse reconstruction."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from project.pipelines.base_step import BaseStep


class ColmapSparseReconstructionStep(BaseStep):
    """Copy from NerfStudio pipeline: COLMAP mapper stage."""

    STEP_NAME = "colmap_sparse_reconstruction"
    OUTPUT_FILES = [
        "0/cameras.bin",
        "0/images.bin",
        "0/points3D.bin",
    ]

    def run(self, force: bool = False) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        if not force and self.check_outputs():
            logging.info("🏗️ COLMAP sparse reconstruction already done - SKIPPED")
            return
        logging.info("   🏗️ Sparse reconstruction")
        distorted_db_path = self.output_root.parent / "database.db"
        mapper_cmd = (
            f"colmap mapper "
            f"--database_path {distorted_db_path} "
            f"--image_path {self.input_root} "
            f"--output_path {self.output_root} "
            f"--Mapper.ba_global_function_tolerance=0.000001 "
            f"--Mapper.tri_ignore_two_view_tracks 0 "
            f"--Mapper.tri_min_angle 1"
        )
        ret_code = subprocess.call(mapper_cmd, shell=True)
        assert ret_code == 0, f"COLMAP mapper failed with code {ret_code}"
