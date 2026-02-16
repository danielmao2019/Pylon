import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import dash
import torch

from data.structures.three_d.camera.camera import Camera
from models.three_d.base import BaseSceneModel
from models.three_d.gspl.layout import build_display
from models.three_d.gspl.loader import load_gspl_model
from models.three_d.gspl.model import GSPLModel
from models.three_d.gspl.render.display import render_display


class GSPLSceneModel(BaseSceneModel):
    """Scene model wrapper for Gaussian Splatting Lightning exports."""

    def _load_model(self) -> GSPLModel:
        ply_path = GSPLSceneModel._resolve_point_cloud_path(self.resolved_path)
        model = load_gspl_model(scene_path=str(ply_path), device=self.device)
        assert isinstance(model, GSPLModel), f"Expected GSPLModel, got {type(model)}"
        return model

    def extract_positions(self) -> torch.Tensor:
        gspl_model = self.model
        assert isinstance(gspl_model, GSPLModel), f"{type(gspl_model)=}"
        return gspl_model.get_xyz

    @staticmethod
    def parse_scene_path(path: str) -> str:
        resolved_path = os.path.abspath(path)
        assert os.path.isdir(
            resolved_path
        ), f"Expected existing directory for gspl scene, got '{path}'"

        cfg_path = os.path.join(resolved_path, 'cfg_args')
        assert os.path.isfile(cfg_path), f"cfg_args not found at '{cfg_path}'"
        BaseSceneModel._load_cfg_args(scene_path=resolved_path)

        point_cloud_dir = Path(resolved_path) / 'point_cloud'
        iteration_dir = GSPLSceneModel._select_iteration_dir(point_cloud_dir)
        ply_path = iteration_dir / 'point_cloud.ply'
        assert ply_path.is_file(), f"point_cloud.ply missing under '{iteration_dir}'"

        cameras_path = Path(resolved_path) / 'cameras.json'
        assert cameras_path.is_file(), f"cameras.json not found at '{cameras_path}'"

        lightning_logs_dir = Path(resolved_path) / 'lightning_logs'
        assert lightning_logs_dir.is_dir(), (
            "GSPL outputs must include a lightning_logs directory for checkpoints "
            f"({lightning_logs_dir})"
        )

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
        # Input validations
        assert isinstance(camera, Camera), f"{type(camera)=}"
        assert isinstance(resolution, tuple), f"{type(resolution)=}"
        assert len(resolution) == 2, f"{len(resolution)=}"
        assert all(isinstance(dim, int) for dim in resolution), f"{resolution=}"
        assert camera_name is None or isinstance(camera_name, str), f"{type(camera_name)=}"
        assert display_cameras is None or isinstance(
            display_cameras, list
        ), f"{type(display_cameras)=}"
        assert display_cameras is None or all(
            isinstance(cam, Camera) for cam in display_cameras
        ), f"{display_cameras=}"
        assert title is None or isinstance(title, str), f"{type(title)=}"
        assert device is None or isinstance(device, torch.device), f"{type(device)=}"

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
    def _select_iteration_dir(point_cloud_dir: Path) -> Path:
        assert (
            point_cloud_dir.is_dir()
        ), f"point_cloud directory missing at {point_cloud_dir}"

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
    def _resolve_point_cloud_path(scene_path: str) -> Path:
        point_cloud_dir = Path(scene_path) / 'point_cloud'
        iteration_dir = GSPLSceneModel._select_iteration_dir(point_cloud_dir)
        ply_path = iteration_dir / 'point_cloud.ply'
        assert ply_path.is_file(), f"point_cloud.ply missing at '{ply_path}'"
        return ply_path
