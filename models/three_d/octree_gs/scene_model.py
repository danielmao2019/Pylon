import os
from argparse import Namespace
from pathlib import Path
from typing import Any, List, Optional, Tuple

import dash
import torch
from dash import html

from data.structures.three_d.camera.camera import Camera
from models.three_d.base import BaseSceneModel
from models.three_d.octree_gs import callbacks as octree_gs_callbacks
from models.three_d.octree_gs import states as octree_gs_states
from models.three_d.octree_gs import styles
from models.three_d.octree_gs.layout import build_display
from models.three_d.octree_gs.loader import load_octree_gs_3dgs
from models.three_d.octree_gs.render.display import render_display


class OctreeGSSceneModel(BaseSceneModel):

    def _load_model(self) -> Any:
        return load_octree_gs_3dgs(self.resolved_path, device=self.device)

    @staticmethod
    def parse_scene_path(path: str) -> str:
        assert os.path.isdir(path)
        assert os.path.isfile(os.path.join(path, "cameras.json"))
        assert os.path.isfile(os.path.join(path, "cfg_args"))
        point_cloud_dir = Path(path) / "point_cloud"
        assert (
            point_cloud_dir.is_dir()
        ), f"OctreeGS requires point_cloud directory under {path}"
        has_allowed_checkpoint = any(
            (point_cloud_dir / f"iteration_{iteration}" / "point_cloud.ply").is_file()
            for iteration in (0, 30000)
        )
        assert has_allowed_checkpoint, (
            "OctreeGS point_cloud directory must include point_cloud.ply under "
            "iteration_0 or iteration_30000"
        )

        cfg_path = os.path.join(path, "cfg_args")
        with open(cfg_path, 'r', encoding='utf-8') as handle:
            cfg_text = handle.read()

        cfg = eval(cfg_text)
        assert isinstance(
            cfg, Namespace
        ), f"cfg_args at '{cfg_path}' did not evaluate to Namespace (got {type(cfg)})"
        assert hasattr(
            cfg, 'base_model'
        ), f"cfg_args at '{cfg_path}' missing base_model"
        base_model = cfg.base_model
        assert isinstance(
            base_model, str
        ), f"cfg_args at '{cfg_path}' base_model must be str, got {type(base_model)}"
        assert (
            base_model == '3dgs'
        ), f"cfg_args at '{cfg_path}' base_model must be '3dgs' for OctreeGS (got {base_model!r})"
        assert hasattr(
            cfg, 'model_config'
        ), f"cfg_args at '{cfg_path}' missing model_config"
        model_config = cfg.model_config
        assert isinstance(
            model_config, dict
        ), f"cfg_args at '{cfg_path}' model_config must be dict, got {type(model_config)}"
        assert (
            'name' in model_config
        ), f"cfg_args at '{cfg_path}' model_config missing 'name' key"
        model_name = model_config['name']
        assert model_name == 'GaussianLoDModel', (
            "cfg_args model_config.name must be 'GaussianLoDModel' for octree_gs "
            f"(got {model_name!r})"
        )

        return os.path.abspath(path)

    @staticmethod
    def extract_scene_name(resolved_path: str) -> str:
        path_obj = Path(resolved_path).resolve()
        base = path_obj.parent.name
        parts = base.split('-')
        if len(parts) >= 6:
            scene_name = '-'.join(parts[:6])
        else:
            scene_name = base
        return scene_name

    @staticmethod
    def infer_data_dir(resolved_path: str) -> Optional[str]:
        return BaseSceneModel._infer_data_dir_from_cfg_args(resolved_path)

    @staticmethod
    def register_callbacks(
        dataset: Any, app: dash.Dash, viewer: Any, **kwargs: Any
    ) -> None:
        octree_gs_callbacks.register_callbacks(
            dataset=dataset,
            app=app,
            viewer=viewer,
        )

    @staticmethod
    def setup_states(app: dash.Dash, **kwargs: Any) -> None:
        assert 'dataset_name' in kwargs, "dataset_name required for octree state setup"
        assert 'scene_name' in kwargs, "scene_name required for octree state setup"
        assert 'method_name' in kwargs, "method_name required for octree state setup"
        dataset_name = kwargs['dataset_name']
        scene_name = kwargs['scene_name']
        method_name = kwargs['method_name']
        octree_gs_states.setup_states(
            app=app,
            scene_model_cls=OctreeGSSceneModel,
            dataset_name=dataset_name,
            scene_name=scene_name,
            method_name=method_name,
        )

    def build_static_container(
        self,
        dataset_name: str,
        scene_name: str,
        method_name: str,
        debugger_enabled: bool,
    ) -> html.Div:
        assert isinstance(dataset_name, str), f"{type(dataset_name)=}"
        assert isinstance(scene_name, str), f"{type(scene_name)=}"
        assert isinstance(method_name, str), f"{type(method_name)=}"
        assert isinstance(debugger_enabled, bool), f"{type(debugger_enabled)=}"
        toggle_button = html.Button(
            'Toggle Debugger',
            id={
                'type': 'octree-debugger-toggle',
                'dataset': dataset_name,
                'scene': scene_name,
                'method': method_name,
            },
            n_clicks=0,
            style=styles.toggle_button_style(),
        )
        status_text = html.Span(
            'Debugger ON' if debugger_enabled else 'Debugger OFF',
            style=styles.status_text_style(debugger_enabled),
        )
        overlay = html.Div(
            [
                toggle_button,
                status_text,
            ],
            style=styles.toggle_overlay_style(),
        )
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
            [overlay, body_placeholder],
            id={
                'type': 'octree-gs-container',
                'dataset': dataset_name,
                'scene': scene_name,
                'method': method_name,
            },
            style=styles.container_style(),
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
        """Render the Octree GS layout with an optional debugger overlay."""
        # Input validation
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
        assert (
            'dataset_name' in kwargs
        ), "dataset_name is required for Octree GS rendering"
        assert 'scene_name' in kwargs, "scene_name is required for Octree GS rendering"
        assert 'method' in kwargs, "method is required for Octree GS rendering"
        assert isinstance(
            kwargs['dataset_name'], str
        ), f"{type(kwargs['dataset_name'])=}"
        assert isinstance(kwargs['scene_name'], str), f"{type(kwargs['scene_name'])=}"
        assert isinstance(kwargs['method'], str), f"{type(kwargs['method'])=}"
        assert (
            'octree_debugger_enabled' in kwargs
        ), "octree_debugger_enabled required in kwargs"
        assert isinstance(
            kwargs['octree_debugger_enabled'], bool
        ), f"octree_debugger_enabled must be bool, got {type(kwargs['octree_debugger_enabled'])=}"
        assert (
            'octree_selected_levels_rgb' in kwargs
        ), "octree_selected_levels_rgb required in kwargs"
        assert (
            'octree_selected_levels_density' in kwargs
        ), "octree_selected_levels_density required in kwargs"

        rgb_levels_raw = kwargs['octree_selected_levels_rgb']
        density_levels_raw = kwargs['octree_selected_levels_density']
        assert rgb_levels_raw is None or isinstance(
            rgb_levels_raw, list
        ), f"{type(rgb_levels_raw)=}"
        assert density_levels_raw is None or isinstance(
            density_levels_raw, list
        ), f"{type(density_levels_raw)=}"
        total_levels = self.model.levels
        assert isinstance(total_levels, int), f"{type(total_levels)=}"
        sanitized_rgb = self._sanitize_levels(
            levels=rgb_levels_raw, total_levels=total_levels
        )
        sanitized_density = self._sanitize_levels(
            levels=density_levels_raw, total_levels=total_levels
        )
        target_camera_name = camera_name if camera_name is not None else camera.name

        render_outputs = render_display(
            scene_model=self,
            camera=camera,
            resolution=resolution,
            dataset_name=kwargs['dataset_name'],
            scene_name=kwargs['scene_name'],
            method_name=kwargs['method'],
            debugger_enabled=kwargs['octree_debugger_enabled'],
            selected_levels_rgb=sanitized_rgb,
            selected_levels_density=sanitized_density,
            camera_name=target_camera_name,
            display_cameras=display_cameras,
            title=title,
            device=device if device is not None else self.device,
        )
        return build_display(render_outputs)

    @staticmethod
    def _sanitize_levels(levels: Optional[List[Any]], total_levels: int) -> List[int]:
        if levels is None:
            return list(range(total_levels))
        assert isinstance(levels, list), f"{type(levels)=}"
        sanitized = sorted(
            {int(level) for level in levels if 0 <= int(level) < total_levels}
        )
        assert sanitized, "Level selection must not be empty"
        return sanitized
