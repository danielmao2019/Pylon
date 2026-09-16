# `models/three_d/point_cloud/` code skeleton

## Code implementation structure

`models/three_d/point_cloud/scene_model.py`

```text
scene_model.py
├── import os
├── from typing import Any, List, Optional, Tuple
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.point_cloud import PointCloud, load_point_cloud
├── from models.three_d.base import BaseSceneModel
├── from models.three_d.point_cloud.layout import build_display
├── from models.three_d.point_cloud.render import render_display
└── class PointCloudSceneModel(BaseSceneModel)
    ├── # The scene model the interactive display drives when the scene is one point cloud file.
    ├── def _load_model(self) -> PointCloud [override]
    │   ├── # Opens the file the scene path resolved to, onto the device the scene model was built for.
    │   ├── calls load_point_cloud(self.resolved_path, device=self.device)
    │   └── return  # the point cloud it loaded
    ├── def extract_positions(self) -> torch.Tensor [override]
    │   ├── # Hands the display the coordinates it draws.
    │   ├── impls point_cloud = self.model
    │   ├── assert point_cloud is a PointCloud
    │   └── return  # the coordinates of point_cloud
    ├── @staticmethod def parse_scene_path(path: str) -> str [override]
    │   ├── # Validates a direct point cloud filepath and resolves it.
    │   ├── assert path names an existing file
    │   └── return  # the absolute path, by os.path.abspath
    ├── @staticmethod def extract_scene_name(resolved_path: str) -> str [override]
    │   ├── # Names the scene by the leading six dash-separated fields of the file's own stem.
    │   ├── impls basename = the final component of resolved_path
    │   ├── impls stem = basename without its extension
    │   ├── impls parts = stem split on dashes
    │   ├── impls scene_name = the first six of parts rejoined on dashes
    │   └── return scene_name
    ├── @staticmethod def infer_data_dir(resolved_path: str) -> Optional[str] [override]
    │   ├── # Takes the directory the point cloud file itself sits in as the scene's data directory.
    │   └── return  # the directory of the absolute resolved_path
    └── def display_render(self, camera: Camera, resolution: Tuple[int, int], camera_name: Optional[str] = None, display_cameras: Optional[List[Camera]] = None, title: Optional[str] = None, device: Optional[torch.device] = None, **kwargs: Any) -> Any [override]
        ├── # Renders this scene through one camera and wraps that render in the component the viewer mounts.
        ├── assert camera is a Camera                             # f"{type(camera)=}"
        ├── assert resolution is a tuple                          # f"{type(resolution)=}"
        ├── assert resolution holds two entries                   # f"{len(resolution)=}"
        ├── assert all(dim is an int for each dim in resolution)  # f"{resolution=}"
        │   └── for each dim in resolution
        │       └── impls whether dim is an int
        ├── assert camera_name is None or a str       # f"{type(camera_name)=}"
        ├── assert display_cameras is None or a list  # f"{type(display_cameras)=}"
        ├── assert display_cameras is None or all(cam is a Camera for each cam in display_cameras)  # f"{display_cameras=}"
        │   └── for each cam in display_cameras
        │       └── impls whether cam is a Camera
        ├── assert title is None or a str            # f"{type(title)=}"
        ├── assert device is None or a torch.device  # f"{type(device)=}"
        ├── impls target_camera_name = camera_name if camera_name is not None, else the camera's own name
        │   ├── if camera_name is not None
        │   │   └── impls camera_name
        │   └── else
        │       └── impls the camera's own name
        ├── calls render_display(scene_model=self, camera=camera, resolution=resolution, camera_name=target_camera_name, display_cameras=display_cameras, title=title, device=device)  # -> render_outputs
        ├── calls build_display(render_outputs)
        └── return  # the display component it built
```
