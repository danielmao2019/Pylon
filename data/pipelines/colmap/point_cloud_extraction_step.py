"""Step that exports the COLMAP sparse point cloud as a PLY file."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

from project.pipelines.base_step import BaseStep


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

SPEC = importlib.util.spec_from_file_location(
    "project.scripts.3_prepare_nerfstudio_data.colmap_utils",
    str(
        REPO_ROOT
        / "project"
        / "scripts"
        / "3_prepare_nerfstudio_data"
        / "colmap_utils.py"
    ),
)
assert (
    SPEC is not None and SPEC.loader is not None
), "Unable to load colmap_utils module from repository root"
_colmap_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_colmap_utils)
read_model = _colmap_utils.read_model
create_ply_from_colmap = _colmap_utils.create_ply_from_colmap


class ColmapPointCloudExtractionStep(BaseStep):
    """Re-export the COLMAP sparse reconstruction into our standard layout."""

    STEP_NAME = "colmap_point_cloud_extraction"
    INPUT_FILES = ["0/points3D.bin"]
    OUTPUT_FILES = ["sparse_pc.ply"]

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        super().__init__(input_root=input_root, output_root=output_root)
        self.model_dir = self.input_root / "0"
        self.output_path = self.output_root / self.OUTPUT_FILES[0]

    def run(self, force: bool = False) -> None:
        self.check_inputs()
        outputs_ready = self.check_outputs()
        if outputs_ready and not force:
            logging.info("🌐 COLMAP sparse point cloud already extracted - SKIPPED")
            return

        logging.info("🌐 Extracting COLMAP sparse point cloud")
        _, _, points3D = read_model(str(self.model_dir))
        ply_path = create_ply_from_colmap(
            "sparse_pc.ply",
            points3D,
            str(self.output_root),
        )
        logging.info("   ✓ Wrote COLMAP sparse point cloud: %s", ply_path)
