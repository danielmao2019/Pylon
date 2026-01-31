"""Pipeline that mirrors the original COLMAP-to-NeRF export script."""

from pathlib import Path
from typing import Any, Dict, Optional

from data.pipelines.base_pipeline import BasePipeline
from data.pipelines.colmap.core.colmap_core_pipeline import ColmapCorePipeline
from data.pipelines.colmap.extract_cameras_step import ColmapExtractCamerasStep
from data.pipelines.colmap.extract_point_cloud_step import ColmapExtractPointCloudStep


class ColmapPipeline(BasePipeline):
    """Sequential pipeline that exports NeRF-ready data from a folder of images."""

    PIPELINE_NAME = "colmap_pipeline"

    def __init__(
        self,
        scene_root: str | Path,
        matcher_cfg: Optional[Dict[str, Any]] = None,
        upright: bool = False,
        camera_mode: str = "OPENCV",
        init_from_dji: bool = False,
        dji_data_root: str | Path | None = None,
        mask_input_root: str | Path | None = None,
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
        assert isinstance(init_from_dji, bool), f"{type(init_from_dji)=}"
        assert (
            not init_from_dji
        ) or dji_data_root is not None, (
            "dji_data_root must be provided when init_from_dji is True"
        )
        assert dji_data_root is None or isinstance(
            dji_data_root, (str, Path)
        ), f"{type(dji_data_root)=}"
        assert mask_input_root is None or isinstance(
            mask_input_root, (str, Path)
        ), f"{type(mask_input_root)=}"

        self.scene_root = Path(scene_root).expanduser().resolve()

        step_configs = [
            {
                "class": ColmapCorePipeline,
                "args": {
                    "scene_root": self.scene_root,
                    "matcher_cfg": matcher_cfg,
                    "upright": upright,
                    "camera_mode": camera_mode,
                    "init_from_dji": init_from_dji,
                    "dji_data_root": dji_data_root,
                    "mask_input_root": mask_input_root,
                },
            },
            {
                "class": ColmapExtractCamerasStep,
                "args": {
                    "input_root": self.scene_root / "undistorted" / "sparse",
                    "output_root": self.scene_root,
                },
            },
            {
                "class": ColmapExtractPointCloudStep,
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
