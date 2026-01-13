"""Step that exports the COLMAP sparse point cloud as a PLY file."""

import logging
from pathlib import Path
from typing import Any, Dict

from data.pipelines.base_step import BaseStep
from utils.io.colmap.load_colmap import (
    create_ply_from_colmap,
    load_model,
)


class ColmapExtractPointCloudStep(BaseStep):
    """Re-export the COLMAP sparse reconstruction into our standard layout."""

    STEP_NAME = "colmap_extract_point_cloud"

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        super().__init__(input_root=input_root, output_root=output_root)
        self.model_dir = self.input_root / "0"
        self.output_path = self.output_root / self.output_files[0]

    def _init_input_files(self) -> None:
        self.input_files = ["0/points3D.bin"]

    def _init_output_files(self) -> None:
        self.output_files = ["sparse_pc.ply"]

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        outputs_ready = self.check_outputs()
        if outputs_ready and not force:
            logging.info("🌐 COLMAP sparse point cloud already extracted - SKIPPED")
            return {}

        logging.info("🌐 Extracting COLMAP sparse point cloud")
        _, _, points3D = load_model(str(self.model_dir))
        ply_path = create_ply_from_colmap(
            "sparse_pc.ply",
            points3D,
            str(self.output_root),
        )
        logging.info("   ✓ Wrote COLMAP sparse point cloud: %s", ply_path)
        return {}
