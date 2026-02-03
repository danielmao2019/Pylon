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
        matcher_cfg: Optional[Dict[str, Any]] = None,
        upright: bool = False,
        camera_mode: str = "OPENCV",
        init_step: Dict[str, Any] | None = None,
        mask_input_root: str | Path | None = None,
        strict: bool = True,
    ) -> None:
        # Input validations
        assert isinstance(scene_root, (str, Path)), f"{type(scene_root)=}"
        assert matcher_cfg is None or isinstance(
            matcher_cfg, dict
        ), f"{type(matcher_cfg)=}"
        assert isinstance(upright, bool), f"{type(upright)=}"
        assert isinstance(camera_mode, str), f"{type(camera_mode)=}"
        assert camera_mode in {
            "SIMPLE_PINHOLE",
            "PINHOLE",
            "OPENCV",
        }, f"{camera_mode=}"
        assert init_step is None or isinstance(
            init_step, (BasePipeline, BaseStep, dict)
        ), f"{type(init_step)=}"
        assert (
            init_step is None or not isinstance(init_step, dict) or "class" in init_step
        ), "init_step must include class"
        assert (
            init_step is None or not isinstance(init_step, dict) or "args" in init_step
        ), "init_step must include args"
        assert (
            init_step is None
            or not isinstance(init_step, dict)
            or isinstance(init_step["args"], dict)
        ), f"{type(init_step['args'])=}"
        assert mask_input_root is None or isinstance(
            mask_input_root, (str, Path)
        ), f"{type(mask_input_root)=}"
        assert isinstance(strict, bool), f"{type(strict)=}"

        self.scene_root = Path(scene_root).expanduser().resolve()
        self.colmap_args = self._get_colmap_args()
        step_configs = self._build_steps(
            matcher_cfg=matcher_cfg,
            upright=upright,
            camera_mode=camera_mode,
            init_step=init_step,
            mask_input_root=mask_input_root,
            strict=strict,
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
        matcher_cfg: Optional[Dict[str, Any]],
        upright: bool,
        camera_mode: str,
        init_step: Dict[str, Any] | None,
        mask_input_root: str | Path | None,
        strict: bool,
    ) -> List[Dict[str, Any]]:
        common_prefix = [
            {
                "class": ColmapFeatureExtractionStep,
                "args": {
                    "scene_root": self.scene_root,
                    "colmap_args": self.colmap_args,
                    "upright": upright,
                    "camera_mode": camera_mode,
                    "mask_input_root": mask_input_root,
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
                    "class": ColmapSparseReconstructionStep,
                    "args": {
                        "scene_root": self.scene_root,
                        "init_model_dir": self.scene_root / "distorted" / "init_model",
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
                    "model_relpath": "distorted/sparse/0",
                },
            },
            {
                "class": ColmapModelTxtExportStep,
                "args": {
                    "scene_root": self.scene_root,
                    "model_relpath": "undistorted/sparse/0",
                },
            },
        ]
        return common_prefix + reconstruction_steps + common_suffix
