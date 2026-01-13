import math
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, List, Optional, Tuple

import dash
import torch

from data.structures.three_d.camera.camera import Camera
from models.three_d.base import BaseSceneModel
from models.three_d.letsgo.layout import build_display
from models.three_d.letsgo.model import LetsGoModel
from models.three_d.letsgo.render.display import render_display


class LetsGoSceneModel(BaseSceneModel):
    """Scene model wrapper for LetsGo LoD Gaussian splatting exports."""

    def _load_model(self) -> Any:
        cfg = BaseSceneModel._load_cfg_args(self.resolved_path)
        self._assert_cfg_fields(cfg=cfg)

        sh_degree = int(cfg.sh_degree)
        assert sh_degree >= 0, f"sh_degree must be non-negative, got {sh_degree}"
        depth_max = float(cfg.depth_max)
        assert depth_max > 0.0, f"depth_max must be positive, got {depth_max}"
        use_lod = bool(cfg.use_lod)
        assert use_lod, "LetsGo checkpoints must be trained with LoD enabled"

        iteration_dir = LetsGoSceneModel._select_iteration_dir(self.resolved_path)
        max_level = LetsGoSceneModel._determine_max_level(iteration_dir)
        beta = math.log(float(max_level + 1))

        model = LetsGoModel(
            sh_degree=sh_degree,
            max_level=max_level,
            depth_max=depth_max,
            beta=beta,
        )
        model.load_ply(str(iteration_dir))
        return model

    @staticmethod
    def parse_scene_path(path: str) -> str:
        resolved_path = os.path.abspath(path)
        assert os.path.isdir(
            resolved_path
        ), f"Expected existing directory for letsgo scene, got '{path}'"

        cfg_path = os.path.join(resolved_path, 'cfg_args')
        assert os.path.isfile(cfg_path), f"cfg_args not found at '{cfg_path}'"
        BaseSceneModel._load_cfg_args(scene_path=resolved_path)

        point_cloud_dir = os.path.join(resolved_path, 'point_cloud')
        assert os.path.isdir(
            point_cloud_dir
        ), f"point_cloud directory missing under '{resolved_path}'"
        iteration_dir = LetsGoSceneModel._select_iteration_dir(resolved_path)
        LetsGoSceneModel._determine_max_level(iteration_dir)

        cameras_path = os.path.join(resolved_path, 'cameras.json')
        assert os.path.isfile(
            cameras_path
        ), f"cameras.json not found under '{resolved_path}'"

        return resolved_path

    @staticmethod
    def extract_scene_name(resolved_path: str) -> str:
        path_obj = Path(resolved_path).resolve()
        base = path_obj.parent.name
        parts = base.split('-')
        if len(parts) >= 6:
            return '-'.join(parts[:6])
        return base

    @staticmethod
    def infer_data_dir(resolved_path: str) -> Optional[str]:
        return BaseSceneModel._infer_data_dir_from_cfg_args(resolved_path)

    @staticmethod
    def register_callbacks(
        dataset: Any, app: dash.Dash, viewer: Any, **kwargs: Any
    ) -> None:
        return None

    @staticmethod
    def setup_states(app: dash.Dash, **kwargs: Any) -> None:
        return None

    def display_render(
        self,
        camera: Camera,
        resolution: Tuple[int, int],
        camera_name: Optional[str] = None,
        display_cameras: Optional[List[Camera]] = None,
        title: Optional[str] = None,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Any:
        assert isinstance(camera, Camera), f"{type(camera)=}"
        target_camera_name = camera_name if camera_name is not None else camera.name
        render_outputs = render_display(
            scene_model=self,
            camera=camera,
            resolution=resolution,
            camera_name=target_camera_name,
            display_cameras=display_cameras,
            title=title,
            device=device,
        )
        return build_display(render_outputs)

    @staticmethod
    def _select_iteration_dir(scene_path: str) -> Path:
        point_cloud_dir = Path(scene_path) / 'point_cloud'
        assert (
            point_cloud_dir.is_dir()
        ), f"point_cloud directory missing at {scene_path}"

        iteration_dirs: List[Tuple[int, Path]] = []
        for entry in point_cloud_dir.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            if not name.startswith('iteration_'):
                continue
            suffix = name.split('iteration_')[-1]
            assert suffix.isdigit(), f"Invalid iteration directory name '{name}'"
            iteration_dirs.append((int(suffix), entry))

        assert (
            iteration_dirs
        ), f"No iteration_* directories found under {point_cloud_dir}"
        iteration_dirs.sort(key=lambda pair: pair[0])
        return iteration_dirs[-1][1]

    @staticmethod
    def _determine_max_level(iteration_dir: Path) -> int:
        level_files = sorted(iteration_dir.glob('level_*.ply'))
        assert level_files, f"No level_*.ply files found under '{iteration_dir}'"

        levels: List[int] = []
        for level_file in level_files:
            stem_parts = level_file.stem.split('_')
            assert (
                len(stem_parts) == 2
            ), f"Unexpected level filename '{level_file.name}' in '{iteration_dir}'"
            levels.append(int(stem_parts[1]))

        max_level = max(levels)
        expected_levels = list(range(max_level + 1))
        observed_levels = sorted(set(levels))
        assert (
            observed_levels == expected_levels
        ), f"Missing level files for letsgo checkpoint: expected {expected_levels}, found {observed_levels}"

        return max_level

    @staticmethod
    def _assert_cfg_fields(cfg: Namespace) -> None:
        actual_fields = set(vars(cfg).keys())
        expected_fields = {
            'data_device',
            'depth_max',
            'depths',
            'eval',
            'images',
            'model_path',
            'resolution',
            'sh_degree',
            'source_path',
            'use_lod',
            'white_background',
        }
        assert (
            actual_fields == expected_fields
        ), f"Unexpected cfg_args fields for LetsGo: {actual_fields}"
        assert isinstance(cfg.data_device, str)
        assert isinstance(cfg.depths, str)
        assert isinstance(cfg.eval, bool)
        assert isinstance(cfg.images, str)
        assert isinstance(cfg.model_path, str)
        assert isinstance(cfg.resolution, int)
        assert isinstance(cfg.sh_degree, int)
        assert isinstance(cfg.source_path, str)
        assert isinstance(cfg.use_lod, bool)
        assert isinstance(cfg.white_background, bool)
