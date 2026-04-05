from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dash
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.mesh.mesh import Mesh
from models.three_d.base import BaseSceneModel
from models.three_d.meshes.callbacks.register import register_callbacks
from models.three_d.meshes.layout.components import build_display
from models.three_d.meshes.render.display import render_display
from models.three_d.meshes.states import setup_states


class BaseMeshesSceneModel(BaseSceneModel):
    """Base mesh scene model with shared loading/rendering logic for OBJ-style meshes."""

    def _load_model(self) -> Mesh:
        """Load one mesh scene as one repo `Mesh` instance.

        Args:
            None.

        Returns:
            Repo mesh data for downstream rendering.
        """
        mesh_dir = Path(self.resolved_path)
        mesh = Mesh.load(path=mesh_dir)
        assert mesh.vertex_color is not None or mesh.uv_texture_map is not None, (
            "Expected mesh scene loading to produce a textured repo `Mesh`. "
            f"{mesh.vertex_color is not None=} {mesh.uv_texture_map is not None=}"
        )
        return mesh

    def extract_positions(self) -> torch.Tensor:
        mesh = self.model
        assert isinstance(mesh, Mesh), f"{type(mesh)=}"
        return mesh.vertices

    @staticmethod
    @abstractmethod
    def parse_scene_path(path: str) -> str:
        """Resolve and validate a mesh path. Subclasses must implement."""
        raise NotImplementedError("parse_scene_path must be implemented by subclasses")

    @staticmethod
    @abstractmethod
    def extract_scene_name(resolved_path: str) -> str:
        """Derive a scene name from a resolved mesh path. Subclasses must implement."""
        raise NotImplementedError(
            "extract_scene_name must be implemented by subclasses"
        )

    @staticmethod
    @abstractmethod
    def infer_data_dir(resolved_path: str) -> Optional[str]:
        """Infer the data directory (e.g., containing transforms.json). Subclasses must implement."""
        raise NotImplementedError("infer_data_dir must be implemented by subclasses")

    @staticmethod
    def register_callbacks(
        dataset: Any, app: dash.Dash, viewer: Any, **kwargs: Any
    ) -> None:
        register_callbacks(
            dataset=dataset,
            app=app,
            viewer=viewer,
        )

    @staticmethod
    def setup_states(app: dash.Dash, **kwargs: Any) -> None:
        setup_states(app=app)

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

    def to(self, device: torch.device) -> "BaseMeshesSceneModel":
        """Move the loaded repo mesh to a target device.

        Args:
            device: Target device for the stored repo mesh.

        Returns:
            This scene model on the target device.
        """

        assert isinstance(device, torch.device), f"{type(device)=}"
        self._ensure_model_loaded()
        assert isinstance(self._model, Mesh), f"{type(self._model)=}"

        self._model = self._model.to(device=device)
        self.device = device
        return self

    def cpu(self) -> "BaseMeshesSceneModel":
        return self.to(torch.device('cpu'))
