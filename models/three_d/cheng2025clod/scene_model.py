import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import dash
import torch
from dash import html

from data.structures.three_d.camera.camera import Camera
from models.three_d.base import BaseSceneModel
from models.three_d.cheng2025clod.loader import load_cheng2025_clod_model
from models.three_d.cheng2025clod.render import render_display
from models.three_d.original_3dgs.styles import styles as original_styles
from models.three_d.original_3dgs.layout import build_display


class Cheng2025CLODSceneModel(BaseSceneModel):
    """Scene model for Cheng et al. 2025 continuous LoD Gaussian splatting."""

    def __init__(
        self,
        scene_path: str,
        device: Optional[torch.device] = None,
        cache: Optional[Any] = None,
        cache_key: Optional[str] = None,
    ) -> None:
        self._virtual_scale: Optional[float] = None
        self._lod_threshold: Optional[float] = None
        self._gaussians_total: Optional[int] = None
        super().__init__(
            scene_path=scene_path,
            device=device if device is not None else torch.device('cuda'),
            cache=cache,
            cache_key=cache_key,
        )

    def _load_model(self) -> Any:
        cfg = BaseSceneModel._load_cfg_args(self.resolved_path)
        sh_degree = int(cfg.sh_degree)
        self._virtual_scale = 1.0
        self._lod_threshold = 0.01

        model = load_cheng2025_clod_model(
            model_dir=self.resolved_path,
            device=self.device,
            iteration=30_000,
            sh_degree=sh_degree,
        )
        total = int(model.get_xyz.shape[0])
        assert total > 0, "Cheng2025CLOD model must contain Gaussians"
        self._gaussians_total = total
        setattr(model, 'gaussians_count', total)
        return model

    @staticmethod
    def parse_scene_path(path: str) -> str:
        # Input validations
        assert isinstance(path, str), f"{type(path)=}"
        assert os.path.isdir(
            path
        ), f"Expected existing directory for Cheng2025CLOD scene, got '{path}'"

        resolved_path = os.path.abspath(path)
        cfg_path = os.path.join(resolved_path, 'cfg_args')
        assert os.path.isfile(cfg_path), f"cfg_args not found at '{cfg_path}'"
        cfg = BaseSceneModel._load_cfg_args(scene_path=resolved_path)
        required_fields = {
            'data_device',
            'depths',
            'eval',
            'images',
            'model_path',
            'resolution',
            'sh_degree',
            'source_path',
            'train_test_exp',
            'white_background',
            'method',
        }
        actual_fields = set(vars(cfg).keys())
        missing_fields = required_fields - actual_fields
        assert not missing_fields, f"cfg_args missing required fields {missing_fields}"
        unexpected_fields = actual_fields - required_fields
        assert (
            not unexpected_fields
        ), f"cfg_args contains unexpected fields {unexpected_fields}"
        assert (
            cfg.method == "cheng2025clod"
        ), f"cfg_args method must be 'cheng2025clod', got {cfg.method!r}"

        ply_path = os.path.join(
            resolved_path,
            'point_cloud',
            'iteration_30000',
            'point_cloud.ply',
        )
        assert os.path.isfile(
            ply_path
        ), f"point_cloud.ply missing for iteration 30000: {ply_path}"

        cameras_path = os.path.join(resolved_path, 'cameras.json')
        assert os.path.isfile(cameras_path), f"cameras.json not found: {cameras_path}"

        return resolved_path

    @staticmethod
    def extract_scene_name(resolved_path: str) -> str:
        base = Path(resolved_path).resolve().name
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

    def build_static_container(
        self,
        dataset_name: str,
        scene_name: str,
        method_name: str,
        debugger_enabled: bool,
    ) -> html.Div:
        # Input validations
        assert isinstance(dataset_name, str), f"{type(dataset_name)=}"
        assert isinstance(scene_name, str), f"{type(scene_name)=}"
        assert isinstance(method_name, str), f"{type(method_name)=}"
        assert isinstance(debugger_enabled, bool), f"{type(debugger_enabled)=}"

        body_placeholder = html.Div(
            [],
            id={
                'type': 'model-body',
                'dataset': dataset_name,
                'scene': scene_name,
                'method': method_name,
            },
        )
        return html.Div(
            [body_placeholder],
            id={
                'type': 'cheng2025clod-container',
                'dataset': dataset_name,
                'scene': scene_name,
                'method': method_name,
            },
            style=original_styles.container_style(),
        )

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
        assert (
            isinstance(resolution, tuple)
            and len(resolution) == 2
            and all(isinstance(v, int) for v in resolution)
        ), f"{resolution=}"
        if camera_name is not None:
            assert isinstance(camera_name, str), f"{type(camera_name)=}"
        if title is not None:
            assert isinstance(title, str), f"{type(title)=}"
        if device is not None:
            assert isinstance(device, torch.device), f"{type(device)=}"
        if display_cameras is not None:
            assert isinstance(display_cameras, list), f"{type(display_cameras)=}"
            assert all(isinstance(cam, Camera) for cam in display_cameras)

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
