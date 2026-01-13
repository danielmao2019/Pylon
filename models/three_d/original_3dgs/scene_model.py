import os
from typing import Any, List, Optional, Tuple

import dash
import torch

from data.structures.three_d.camera.camera import Camera
from models.three_d.base import BaseSceneModel
from models.three_d.original_3dgs import callbacks as original_3dgs_callbacks
from models.three_d.original_3dgs import states as original_3dgs_states
from models.three_d.original_3dgs.layout import build_display
from models.three_d.original_3dgs.loader import load_3dgs_model_original
from models.three_d.original_3dgs.render.display import render_display


class Original3DGSSceneModel(BaseSceneModel):
    def _load_model(self) -> Any:
        return load_3dgs_model_original(self.resolved_path, device=self.device)

    @staticmethod
    def parse_scene_path(path: str) -> str:
        # Input validations
        assert isinstance(path, str), f"{type(path)=}"
        assert os.path.isdir(
            path
        ), f"Expected existing directory for Original3DGS scene, got '{path}'"

        resolved_path = os.path.abspath(path)
        expected_files = [
            "point_cloud/iteration_30000/point_cloud.ply",
            "cameras.json",
            "cfg_args",
        ]
        assert all(
            os.path.isfile(os.path.join(resolved_path, name)) for name in expected_files
        ), f"Path does not contain expected 3DGS-original files: {expected_files}"

        cfg = BaseSceneModel._load_cfg_args(scene_path=resolved_path)
        expected_fields = {
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
        }
        actual_fields = set(vars(cfg).keys())
        assert (
            actual_fields == expected_fields
        ), f"Unexpected cfg_args fields for original 3DGS: {actual_fields}"

        assert isinstance(cfg.data_device, str), "cfg.data_device must be str"
        assert isinstance(cfg.depths, str), "cfg.depths must be str"
        assert isinstance(cfg.eval, bool), "cfg.eval must be bool"
        assert isinstance(cfg.images, str), "cfg.images must be str"
        assert isinstance(cfg.model_path, str), "cfg.model_path must be str"
        assert isinstance(cfg.resolution, int), "cfg.resolution must be int"
        assert isinstance(cfg.sh_degree, int), "cfg.sh_degree must be int"
        assert isinstance(cfg.source_path, str), "cfg.source_path must be str"
        assert isinstance(cfg.train_test_exp, bool), "cfg.train_test_exp must be bool"
        assert isinstance(
            cfg.white_background, bool
        ), "cfg.white_background must be bool"

        return resolved_path

    @staticmethod
    def extract_scene_name(resolved_path: str) -> str:
        base = os.path.basename(os.path.normpath(resolved_path))
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
        original_3dgs_callbacks.register_callbacks(
            dataset=dataset,
            app=app,
            viewer=viewer,
        )

    @staticmethod
    def setup_states(app: dash.Dash, **kwargs: Any) -> None:
        original_3dgs_states.setup_states(app=app)

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
