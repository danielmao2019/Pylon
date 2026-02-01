from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dash
import torch
from pytorch3d.structures import Meshes

from data.structures.three_d.camera.camera import Camera
from models.three_d.base import BaseSceneModel
from models.three_d.meshes.callbacks.register import register_callbacks
from models.three_d.meshes.layout.components import build_display
from models.three_d.meshes.loader import load_meshes
from models.three_d.meshes.render.display import render_display
from models.three_d.meshes.states import setup_states


class BaseMeshesSceneModel(BaseSceneModel):
    """Base mesh scene model with shared loading/rendering logic for OBJ-style meshes."""

    def _load_model(self) -> Meshes:
        mesh_dir = Path(self.resolved_path)
        merged_mesh = load_meshes(mesh_dir, device=self.device)
        return merged_mesh

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
        assert isinstance(device, torch.device)
        self._ensure_model_loaded()
        assert isinstance(self._model, Meshes)

        verts_target = [verts.to(device=device) for verts in self._model.verts_list()]
        faces_target = [faces.to(device=device) for faces in self._model.faces_list()]
        target_mesh = Meshes(verts=verts_target, faces=faces_target)
        for tensor_attr in self._model._INTERNAL_TENSORS:
            tensor_value = getattr(self._model, tensor_attr)
            if torch.is_tensor(tensor_value):
                setattr(target_mesh, tensor_attr, tensor_value.to(device=device))
        if self._model.textures is not None:
            target_mesh.textures = self._model.textures.to(device=device)

        self._model = target_mesh
        self.device = device
        return self

    def cpu(self) -> "BaseMeshesSceneModel":
        return self.to(torch.device('cpu'))
