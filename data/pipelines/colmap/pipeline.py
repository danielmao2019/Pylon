"""Pipeline that mirrors the original COLMAP-to-NeRF export script."""

from pathlib import Path
from typing import Any, Dict, Optional

from data.pipelines.base_pipeline import BasePipeline
from data.pipelines.base_step import BaseStep
from data.pipelines.colmap.convert_colmap_to_nerfstudio_step import (
    ColmapConvertToNerfstudioStep,
)
from data.pipelines.colmap.core.colmap_core_pipeline import ColmapCorePipeline


class ColmapPipeline(BasePipeline):
    """Sequential pipeline that exports NeRF-ready data from a folder of images."""

    PIPELINE_NAME = "colmap_pipeline"

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

        step_configs = [
            {
                "class": ColmapCorePipeline,
                "args": {
                    "scene_root": self.scene_root,
                    "extractor_cfg": extractor_cfg,
                    "matcher_cfg": matcher_cfg,
                    "reconstruction_cfg": reconstruction_cfg,
                },
            },
            {
                "class": ColmapConvertToNerfstudioStep,
                "args": {
                    "input_root": self.scene_root / "undistorted" / "sparse",
                    "output_root": self.scene_root,
                },
            },
        ]
        super().__init__(
            step_configs=step_configs,
            input_root=self.scene_root,
            output_root=self.scene_root,
        )
