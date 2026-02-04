"""Nested pipeline that groups all COLMAP command invocations."""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.pipelines.base_pipeline import BasePipeline
from data.pipelines.base_step import BaseStep
from data.pipelines.colmap.core.feature_extraction_step import (
    ColmapFeatureExtractionStep,
)
from data.pipelines.colmap.core.feature_matching_step import (
    ColmapFeatureMatchingStep,
)
from data.pipelines.colmap.core.image_undistortion_step import (
    ColmapImageUndistortionStep,
)
from data.pipelines.colmap.core.init.bundle_adjustment_step import (
    ColmapBundleAdjustmentStep,
)
from data.pipelines.colmap.core.init.point_triangulation_step import (
    ColmapPointTriangulationStep,
)
from data.pipelines.colmap.core.model_txt_export_step import (
    ColmapModelTxtExportStep,
)
from data.pipelines.colmap.core.sparse_reconstruction_step import (
    ColmapSparseReconstructionStep,
)


class ColmapCorePipeline(BasePipeline):
    """Encapsulates the COLMAP feature/matcher/mapper/undistortion sequence."""

    PIPELINE_NAME = "colmap_core_pipeline"

    def __init__(
        self,
        scene_root: str | Path,
        extractor_cfg: Dict[str, Any],
        matcher_cfg: Optional[Dict[str, Any]] = None,
        reconstruction_cfg: Dict[str, Any] | None = None,
    ) -> None:
        # Input validations
        assert isinstance(scene_root, (str, Path)), f"{type(scene_root)=}"
        assert isinstance(extractor_cfg, dict), f"{type(extractor_cfg)=}"
        assert extractor_cfg.keys() <= {
            "upright",
            "camera_mode",
            "mask_input_root",
        }, "Invalid extractor_cfg keys"
        assert "upright" in extractor_cfg, "extractor_cfg missing upright"
        assert "camera_mode" in extractor_cfg, "extractor_cfg missing camera_mode"
        assert (
            "mask_input_root" in extractor_cfg
        ), "extractor_cfg missing mask_input_root"
        assert isinstance(
            extractor_cfg["upright"], bool
        ), f"{type(extractor_cfg['upright'])=}"
        assert isinstance(
            extractor_cfg["camera_mode"], str
        ), f"{type(extractor_cfg['camera_mode'])=}"
        assert extractor_cfg["camera_mode"] in {
            "SIMPLE_PINHOLE",
            "PINHOLE",
            "OPENCV",
        }, f"{extractor_cfg['camera_mode']=}"
        assert extractor_cfg["mask_input_root"] is None or isinstance(
            extractor_cfg["mask_input_root"], (str, Path)
        ), f"{type(extractor_cfg['mask_input_root'])=}"
        assert matcher_cfg is None or isinstance(
            matcher_cfg, dict
        ), f"{type(matcher_cfg)=}"
        assert reconstruction_cfg is not None and isinstance(
            reconstruction_cfg, dict
        ), f"{type(reconstruction_cfg)=}"
        assert reconstruction_cfg.keys() <= {
            "init_step",
            "strict",
        }, "Invalid reconstruction_cfg keys"
        assert "init_step" in reconstruction_cfg, "reconstruction_cfg missing init_step"
        assert "strict" in reconstruction_cfg, "reconstruction_cfg missing strict"
        assert isinstance(
            reconstruction_cfg["strict"], bool
        ), f"{type(reconstruction_cfg['strict'])=}"
        assert reconstruction_cfg["init_step"] is None or isinstance(
            reconstruction_cfg["init_step"], (BasePipeline, BaseStep, dict)
        ), f"{type(reconstruction_cfg['init_step'])=}"
        assert (
            reconstruction_cfg["init_step"] is None
            or not isinstance(reconstruction_cfg["init_step"], dict)
            or "class" in reconstruction_cfg["init_step"]
        ), "init_step must include class"
        assert (
            reconstruction_cfg["init_step"] is None
            or not isinstance(reconstruction_cfg["init_step"], dict)
            or "args" in reconstruction_cfg["init_step"]
        ), "init_step must include args"
        assert (
            reconstruction_cfg["init_step"] is None
            or not isinstance(reconstruction_cfg["init_step"], dict)
            or isinstance(reconstruction_cfg["init_step"]["args"], dict)
        ), f"{type(reconstruction_cfg['init_step']['args'])=}"

        self.scene_root = Path(scene_root).expanduser().resolve()
        self.colmap_args = self._get_colmap_args()
        step_configs = self._build_steps(
            extractor_cfg=extractor_cfg,
            matcher_cfg=matcher_cfg,
            reconstruction_cfg=reconstruction_cfg,
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
                "mask_path": "--ImageReader.mask_path",
            }
        return {
            "version": version,
            "feature_use_gpu": "--SiftExtraction.use_gpu",
            "matching_use_gpu": "--SiftMatching.use_gpu",
            "guided_matching": "--SiftMatching.guided_matching",
            "upright": "--SiftExtraction.upright",
            "mask_path": "--ImageReader.mask_path",
        }

    def _build_steps(
        self,
        extractor_cfg: Dict[str, Any],
        matcher_cfg: Optional[Dict[str, Any]],
        reconstruction_cfg: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        init_step = reconstruction_cfg["init_step"]
        strict = reconstruction_cfg["strict"]
        common_prefix = [
            {
                "class": ColmapFeatureExtractionStep,
                "args": {
                    "scene_root": self.scene_root,
                    "colmap_args": self.colmap_args,
                    "extractor_cfg": extractor_cfg,
                },
            },
            {
                "class": ColmapFeatureMatchingStep,
                "args": {
                    "scene_root": self.scene_root,
                    "colmap_args": self.colmap_args,
                    "matcher_cfg": matcher_cfg,
                },
            },
        ]
        if init_step is None:
            reconstruction_steps = [
                {
                    "class": ColmapSparseReconstructionStep,
                    "args": {
                        "scene_root": self.scene_root,
                        "strict": strict,
                    },
                },
            ]
        else:
            reconstruction_steps = [
                init_step,
                {
                    "class": ColmapPointTriangulationStep,
                    "args": {
                        "scene_root": self.scene_root,
                    },
                },
                {
                    "class": ColmapBundleAdjustmentStep,
                    "args": {
                        "scene_root": self.scene_root,
                    },
                },
                {
                    "class": ColmapSparseReconstructionStep,
                    "args": {
                        "scene_root": self.scene_root,
                        "use_init_model": True,
                        "strict": strict,
                    },
                },
            ]
        common_suffix = [
            {
                "class": ColmapImageUndistortionStep,
                "args": {"scene_root": self.scene_root},
            },
            {
                "class": ColmapModelTxtExportStep,
                "args": {
                    "scene_root": self.scene_root,
                    "model_relpath": "distorted/sparse",
                },
            },
            {
                "class": ColmapModelTxtExportStep,
                "args": {
                    "scene_root": self.scene_root,
                    "model_relpath": "undistorted/sparse",
                },
            },
        ]
        return common_prefix + reconstruction_steps + common_suffix
