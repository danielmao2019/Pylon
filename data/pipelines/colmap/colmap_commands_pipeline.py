"""Nested pipeline that groups all COLMAP command invocations."""

from __future__ import annotations

from pathlib import Path

from data.pipelines.base_pipeline import BasePipeline
from data.pipelines.colmap.feature_extraction_step import ColmapFeatureExtractionStep
from data.pipelines.colmap.feature_matching_step import ColmapFeatureMatchingStep
from data.pipelines.colmap.image_undistortion_step import (
    ColmapImageUndistortionStep,
)
from data.pipelines.colmap.sparse_reconstruction_step import (
    ColmapSparseReconstructionStep,
)


class ColmapCommandsPipeline(BasePipeline):
    """Encapsulates the COLMAP feature/matcher/mapper/undistortion sequence."""

    STEP_NAME = "colmap_commands_pipeline"

    def __init__(self, scene_root: str | Path) -> None:
        self.scene_root = Path(scene_root).expanduser().resolve()
        steps = [
            ColmapFeatureExtractionStep(
                input_root=self.scene_root / "input",
                output_root=self.scene_root / "distorted",
            ),
            ColmapFeatureMatchingStep(
                input_root=self.scene_root / "distorted",
                output_root=self.scene_root / "distorted",
            ),
            ColmapSparseReconstructionStep(
                input_root=self.scene_root / "input",
                output_root=self.scene_root / "distorted" / "sparse",
            ),
            ColmapImageUndistortionStep(
                input_root=self.scene_root / "input",
                output_root=self.scene_root,
            ),
        ]
        super().__init__(
            steps=steps,
            input_root=self.scene_root,
            output_root=self.scene_root,
        )
