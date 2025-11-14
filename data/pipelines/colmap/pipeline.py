"""Pipeline that mirrors the original COLMAP-to-NeRF export script."""

from __future__ import annotations

from pathlib import Path

from data.pipelines.base_pipeline import BasePipeline
from data.pipelines.colmap.cleanup_sparse_step import ColmapSparseCleanupStep
from data.pipelines.colmap.colmap_commands_pipeline import ColmapCommandsPipeline
from data.pipelines.colmap.point_cloud_extraction_step import (
    ColmapPointCloudExtractionStep,
)
from data.pipelines.colmap.prepare_inputs_step import PrepareColmapInputsStep


class ColmapPipeline(BasePipeline):
    """Sequential pipeline that exports NeRF-ready data from a folder of images."""

    STEP_NAME = "colmap_pipeline"

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        self.source_images_dir = Path(input_root).expanduser().resolve()
        self.scene_root = Path(output_root).expanduser().resolve()

        steps = [
            PrepareColmapInputsStep(
                input_root=self.source_images_dir,
                output_root=self.scene_root / "input",
            ),
            ColmapCommandsPipeline(scene_root=self.scene_root),
            ColmapSparseCleanupStep(
                input_root=self.scene_root / "sparse",
                output_root=self.scene_root / "undistorted" / "sparse",
            ),
            ColmapPointCloudExtractionStep(
                input_root=self.scene_root / "undistorted" / "sparse",
                output_root=self.scene_root,
            ),
        ]
        super().__init__(
            steps=steps,
            input_root=self.source_images_dir,
            output_root=self.scene_root,
        )
