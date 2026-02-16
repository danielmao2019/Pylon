import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import dash
import torch
from dash import html

from data.structures.three_d.camera.camera import Camera
from models.three_d.base import BaseSceneModel
from models.three_d.lapis_gs import callbacks as lapis_gs_callbacks
from models.three_d.lapis_gs import states as lapis_gs_states
from models.three_d.lapis_gs import styles
from models.three_d.lapis_gs.layout import build_display
from models.three_d.lapis_gs.loader import load_lapis_gs
from models.three_d.original_3dgs.loader import GaussianModel
from models.three_d.lapis_gs.render.display import render_display


class LapisGSSceneModel(BaseSceneModel):

    def _load_model(self) -> Any:
        return load_lapis_gs(self.resolved_path, device=self.device)

    def extract_positions(self) -> torch.Tensor:
        gaussian_model = self.model
        assert isinstance(gaussian_model, GaussianModel), f"{type(gaussian_model)=}"
        return gaussian_model.get_xyz

    @staticmethod
    def parse_scene_path(path: str) -> str:
        assert os.path.isdir(
            path
        ), f"Expected existing directory for lapis_gs, got '{path}'"
        res_suffixes = ['_res1', '_res2', '_res4', '_res8']
        required_files = [
            "point_cloud/iteration_30000/point_cloud.ply",
            "cameras.json",
            "cfg_args",
        ]

        for res_suffix in res_suffixes:
            matching_dirs = [
                entry
                for entry in os.listdir(path)
                if entry.endswith(res_suffix)
                and os.path.isdir(os.path.join(path, entry))
            ]
            assert (
                len(matching_dirs) == 1
            ), f"Expected exactly one subdirectory ending with '{res_suffix}' under '{path}'"
            res_dir = matching_dirs[0]
            res_path = os.path.join(path, res_dir)
            assert all(
                os.path.isfile(os.path.join(res_path, expected))
                for expected in required_files
            ), f"Subdirectory '{res_dir}' under '{path}' is missing required LapisGS files"

        return os.path.abspath(path)

    @staticmethod
    def extract_scene_name(resolved_path: str) -> str:
        path_obj = Path(resolved_path).resolve()
        scene_dir = path_obj.parent
        base = scene_dir.name
        parts = base.split('-')
        if len(parts) >= 6:
            scene_name = '-'.join(parts[:6])
        else:
            scene_name = base
        return scene_name

    @staticmethod
    def infer_data_dir(resolved_path: str) -> Optional[str]:
        res1_dirs = [
            entry
            for entry in os.listdir(resolved_path)
            if entry.endswith('_res1')
            and os.path.isdir(os.path.join(resolved_path, entry))
        ]
        assert (
            len(res1_dirs) == 1
        ), f"Expected exactly one '_res1' subdirectory under '{resolved_path}'"
        res1_path = os.path.join(resolved_path, res1_dirs[0])
        return BaseSceneModel._infer_data_dir_from_cfg_args(res1_path)

    @staticmethod
    def register_callbacks(
        dataset: Any, app: dash.Dash, viewer: Any, **kwargs: Any
    ) -> None:
        lapis_gs_callbacks.register_callbacks(
            dataset=dataset,
            app=app,
            viewer=viewer,
        )

    @staticmethod
    def setup_states(app: dash.Dash, **kwargs: Any) -> None:
        assert 'dataset_name' in kwargs, "dataset_name required for LapisGS state setup"
        assert 'scene_name' in kwargs, "scene_name required for LapisGS state setup"
        assert 'method_name' in kwargs, "method_name required for LapisGS state setup"
        dataset_name = kwargs['dataset_name']
        scene_name = kwargs['scene_name']
        method_name = kwargs['method_name']
        lapis_gs_states.setup_states(
            app=app,
            scene_model_cls=LapisGSSceneModel,
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
                'type': 'lapis-debugger-toggle',
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
                'type': 'lapis-gs-container',
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
        """Render the LapisGS layout with an optional debugger overlay."""
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
        assert {
            'dataset_name',
            'scene_name',
            'method',
            'lapis_debugger_enabled',
            'lapis_selected_layers_rgb',
            'lapis_selected_layers_density',
        }.issubset(kwargs), (
            f"Missing required kwargs: "
            f"{ {'dataset_name','scene_name','method','lapis_debugger_enabled','lapis_selected_layers_rgb','lapis_selected_layers_density'} - set(kwargs) }"
        )
        assert isinstance(
            kwargs['dataset_name'], str
        ), f"{type(kwargs['dataset_name'])=}"
        assert isinstance(kwargs['scene_name'], str), f"{type(kwargs['scene_name'])=}"
        assert isinstance(kwargs['method'], str), f"{type(kwargs['method'])=}"
        assert isinstance(
            kwargs['lapis_debugger_enabled'], bool
        ), f"lapis_debugger_enabled must be bool, got {type(kwargs['lapis_debugger_enabled'])=}"
        assert kwargs['lapis_selected_layers_rgb'] is None or isinstance(
            kwargs['lapis_selected_layers_rgb'], list
        ), f"{type(kwargs['lapis_selected_layers_rgb'])=}"
        assert kwargs['lapis_selected_layers_density'] is None or isinstance(
            kwargs['lapis_selected_layers_density'], list
        ), f"{type(kwargs['lapis_selected_layers_density'])=}"

        rgb_layers_raw = kwargs['lapis_selected_layers_rgb']
        density_layers_raw = kwargs['lapis_selected_layers_density']
        total_layers = len(self.model.split_points) - 1
        assert total_layers > 0, "LapisGS model must have at least one layer"
        sanitized_rgb = self._sanitize_layers(rgb_layers_raw, total_layers)
        sanitized_density = self._sanitize_layers(density_layers_raw, total_layers)
        target_camera_name = camera_name if camera_name is not None else camera.name

        render_outputs = render_display(
            scene_model=self,
            camera=camera,
            resolution=resolution,
            dataset_name=kwargs['dataset_name'],
            scene_name=kwargs['scene_name'],
            method_name=kwargs['method'],
            debugger_enabled=kwargs['lapis_debugger_enabled'],
            selected_layers_rgb=sanitized_rgb,
            selected_layers_density=sanitized_density,
            camera_name=target_camera_name,
            display_cameras=display_cameras,
            title=title,
            device=device if device is not None else self.device,
        )
        return build_display(render_outputs)

    @staticmethod
    def _sanitize_layers(values: Optional[List[Any]], num_layers: int) -> List[int]:
        if values is None:
            return list(range(num_layers))
        assert isinstance(values, list), f"{type(values)=}"
        indices = sorted({int(val) for val in values if 0 <= int(val) < num_layers})
        assert indices, "Layer selection must not be empty"
        return indices
