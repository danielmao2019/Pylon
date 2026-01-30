"""Nested pipeline that groups all COLMAP command invocations."""

import subprocess
from pathlib import Path
from typing import Dict

from data.pipelines.base_pipeline import BasePipeline
from data.pipelines.colmap.feature_extraction_step import ColmapFeatureExtractionStep
from data.pipelines.colmap.feature_matching_step import ColmapFeatureMatchingStep
from data.pipelines.colmap.image_undistortion_step import (
    ColmapImageUndistortionStep,
)
from data.pipelines.colmap.sparse_reconstruction_step import (
    ColmapSparseReconstructionStep,
)
from data.pipelines.colmap.init_from_dji_step import ColmapInitFromDJIStep
from data.pipelines.colmap.point_triangulation_step import (
    ColmapPointTriangulationStep,
)
from data.pipelines.colmap.bundle_adjustment_step import (
    ColmapBundleAdjustmentStep,
)


class ColmapCommandsPipeline(BasePipeline):
    """Encapsulates the COLMAP feature/matcher/mapper/undistortion sequence."""

    PIPELINE_NAME = "colmap_commands_pipeline"

    def __init__(
        self,
        scene_root: str | Path,
        sequential_matching_overlap: int | None = None,
        upright: bool = False,
        init_from_dji: bool = False,
        dji_data_root: str | Path | None = None,
    ) -> None:
        self.scene_root = Path(scene_root).expanduser().resolve()
        if sequential_matching_overlap is not None:
            assert (
                sequential_matching_overlap > 0
            ), "sequential_matching_overlap must be positive"
        self.colmap_args = self._get_colmap_args()
        step_configs = self._build_steps(
            sequential_matching_overlap=sequential_matching_overlap,
            upright=upright,
            init_from_dji=init_from_dji,
            dji_data_root=dji_data_root,
        )
        super().__init__(
            step_configs=step_configs,
            input_root=self.scene_root,
            output_root=self.scene_root,
        )

    def _get_colmap_args(self) -> Dict[str, str]:
        base_help = subprocess.run(
            ["colmap", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        version_line = ""
        for line in base_help.stdout.splitlines():
            if line.startswith("COLMAP"):
                version_line = line
                break
        assert version_line, "COLMAP --help output missing version header"
        tokens = version_line.split()
        assert len(tokens) >= 2, f"Unexpected version line: {version_line}"
        version = tokens[1]
        if version == "3.13.0.dev0":
            return {
                "version": version,
                "feature_use_gpu": "--FeatureExtraction.use_gpu",
                "matching_use_gpu": "--FeatureMatching.use_gpu",
                "guided_matching": "--FeatureMatching.guided_matching",
                "upright": "--SiftExtraction.upright",
            }
        return {
            "version": version,
            "feature_use_gpu": "--SiftExtraction.use_gpu",
            "matching_use_gpu": "--SiftMatching.use_gpu",
            "guided_matching": "--SiftMatching.guided_matching",
            "upright": "--SiftExtraction.upright",
        }

    def _build_steps(
        self,
        sequential_matching_overlap: int | None,
        upright: bool,
        init_from_dji: bool,
        dji_data_root: str | Path | None,
    ) -> list[dict[str, object]]:
        common_prefix = [
            {
                "class": ColmapFeatureExtractionStep,
                "args": {
                    "scene_root": self.scene_root,
                    "colmap_args": self.colmap_args,
                    "upright": upright,
                },
            },
            {
                "class": ColmapFeatureMatchingStep,
                "args": {
                    "scene_root": self.scene_root,
                    "colmap_args": self.colmap_args,
                    "sequential_overlap": sequential_matching_overlap,
                },
            },
        ]
        if not init_from_dji:
            return common_prefix + [
                {
                    "class": ColmapSparseReconstructionStep,
                    "args": {"scene_root": self.scene_root},
                },
                {
                    "class": ColmapImageUndistortionStep,
                    "args": {"scene_root": self.scene_root},
                },
            ]
        assert (
            dji_data_root is not None
        ), "dji_data_root must be provided when init_from_dji is True"
        return common_prefix + [
            {
                "class": ColmapInitFromDJIStep,
                "args": {
                    "scene_root": self.scene_root,
                    "dji_data_root": dji_data_root,
                },
            },
            {
                "class": ColmapPointTriangulationStep,
                "args": {"scene_root": self.scene_root},
            },
            {
                "class": ColmapBundleAdjustmentStep,
                "args": {"scene_root": self.scene_root},
            },
            {
                "class": ColmapImageUndistortionStep,
                "args": {"scene_root": self.scene_root},
            },
        ]
