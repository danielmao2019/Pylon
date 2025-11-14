"""Step that moves undistorted sparse outputs to the IVISION layout."""

from __future__ import annotations

import logging
from pathlib import Path

from project.pipelines.base_step import BaseStep


class ColmapSparseCleanupStep(BaseStep):
    """Copy from NerfStudio pipeline: move sparse dir into undistorted tree."""

    STEP_NAME = "colmap_sparse_cleanup"
    OUTPUT_FILES = [
        "0/cameras.bin",
        "0/images.bin",
        "0/points3D.bin",
    ]

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        super().__init__(input_root=input_root, output_root=output_root)
        self.final_sparse_dir = self.output_root / "0"

    def run(self, force: bool = False) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        if not force and self.check_outputs():
            logging.info("🧹 COLMAP sparse cleanup already done - SKIPPED")
            return
        logging.info("   🧹 Moving sparse outputs")
        self.final_sparse_dir.mkdir(parents=True, exist_ok=True)
        if self.input_root.exists():
            for sparse_file in self.input_root.iterdir():
                if sparse_file.is_file():
                    dest_file = self.final_sparse_dir / sparse_file.name
                    sparse_file.replace(dest_file)
            try:
                self.input_root.rmdir()
            except OSError:
                pass
