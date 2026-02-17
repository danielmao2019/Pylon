import os
from typing import Any, List, Optional, Tuple

import dash
import torch

from data.structures.three_d.camera.camera import Camera
from models.three_d.base import BaseSceneModel
from models.three_d.two_dgs import callbacks as two_dgs_callbacks
from models.three_d.two_dgs import states as two_dgs_states
from models.three_d.two_dgs.layout import build_display
from models.three_d.two_dgs.loader import load_2dgs_model
from models.three_d.two_dgs.model import GaussianModel
from models.three_d.two_dgs.render.display import render_display


class TwoDGSSceneModel(BaseSceneModel):

    def _load_model(self) -> Any:
        return load_2dgs_model(self.resolved_path)

    def extract_positions(self) -> torch.Tensor:
        gaussian_model = self.model
        assert isinstance(gaussian_model, GaussianModel), f"{type(gaussian_model)=}"
        return gaussian_model.get_xyz

    @staticmethod
    def parse_scene_path(path: str) -> str:
        assert os.path.isdir(path)
        resolved_path = os.path.abspath(path)
        expected_files = [
            "point_cloud/iteration_30000/point_cloud.ply",
            "cameras.json",
            "cfg_args",
            "input.ply",
        ]
        assert all(
            os.path.isfile(os.path.join(resolved_path, f)) for f in expected_files
        ), f"Path does not contain expected 2DGS files: {expected_files}"

        cfg = BaseSceneModel._load_cfg_args(scene_path=resolved_path)
        expected_fields = {
            'data_device',
            'eval',
            'images',
            'model_path',
            'render_items',
            'resolution',
            'sh_degree',
            'source_path',
            'white_background',
        }
        actual_fields = set(vars(cfg).keys())
        assert (
            actual_fields == expected_fields
        ), f"Unexpected cfg_args fields for 2DGS: {actual_fields}"
        assert isinstance(cfg.render_items, list), "cfg.render_items must be list"
        assert cfg.render_items, "cfg.render_items must be non-empty"
        assert all(
            isinstance(item, str) for item in cfg.render_items
        ), "cfg.render_items must contain strings"

        return resolved_path

    @staticmethod
    def extract_scene_name(resolved_path: str) -> str:
        base = os.path.basename(os.path.normpath(resolved_path))
        parts = base.split('-')
        assert len(parts) >= 6, (
            f"Invalid 2DGS scene directory name: '{base}'. "
            "Expected at least 6 dash-separated parts like 'Week-XX-DDD-MMM-DD-YYYY'"
        )
        scene_name = '-'.join(parts[:6])
        return scene_name

    @staticmethod
    def infer_data_dir(resolved_path: str) -> Optional[str]:
        return BaseSceneModel._infer_data_dir_from_cfg_args(resolved_path)

    @staticmethod
    def register_callbacks(
        dataset: Any, app: dash.Dash, viewer: Any, **kwargs: Any
    ) -> None:
        two_dgs_callbacks.register_callbacks(
            dataset=dataset,
            app=app,
            viewer=viewer,
        )

    @staticmethod
    def setup_states(app: dash.Dash, **kwargs: Any) -> None:
        two_dgs_states.setup_states(app=app)

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
        assert camera_name is None or isinstance(
            camera_name, str
        ), f"{type(camera_name)=}"
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
