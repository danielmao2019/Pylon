import os
from typing import Any, List, Optional, Tuple

import dash
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.point_cloud import PointCloud, load_point_cloud
from models.three_d.base import BaseSceneModel
from models.three_d.point_cloud import callbacks as point_cloud_callbacks
from models.three_d.point_cloud import states as point_cloud_states
from models.three_d.point_cloud.layout import build_display
from models.three_d.point_cloud.render import render_display


class PointCloudSceneModel(BaseSceneModel):

    def _load_model(self) -> PointCloud:
        return load_point_cloud(self.resolved_path, device=self.device)

    def extract_positions(self) -> torch.Tensor:
        point_cloud = self.model
        assert isinstance(point_cloud, PointCloud), f"{type(point_cloud)=}"
        return point_cloud.xyz

    @staticmethod
    def parse_scene_path(path: str) -> str:
        """Validate a direct point cloud filepath and return it resolved."""
        assert os.path.isfile(path), f"Point cloud path must be a file: {path}"
        return os.path.abspath(path)

    @staticmethod
    def extract_scene_name(resolved_path: str) -> str:
        basename = os.path.basename(resolved_path)
        stem, _ = os.path.splitext(basename)
        parts = stem.split('-')
        scene_name = '-'.join(parts[:6])
        return scene_name

    @staticmethod
    def infer_data_dir(resolved_path: str) -> Optional[str]:
        return os.path.dirname(os.path.abspath(resolved_path))

    @staticmethod
    def register_callbacks(
        dataset: Any, app: dash.Dash, viewer: Any, **kwargs: Any
    ) -> None:
        point_cloud_callbacks.register_callbacks(
            dataset=dataset,
            app=app,
            viewer=viewer,
        )

    @staticmethod
    def setup_states(app: dash.Dash, **kwargs: Any) -> None:
        point_cloud_states.setup_states(app=app)

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
        """Render and display point cloud scene."""
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
