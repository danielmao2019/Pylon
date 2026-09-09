# `models/three_d/point_cloud/` code skeleton

## Code implementation structure

`models/three_d/point_cloud/scene_model.py`

```text
scene_model.py
├── import os
├── import torch
├── from data.structures.three_d.point_cloud import PointCloud, load_point_cloud
├── from models.three_d.base import BaseSceneModel
└── class PointCloudSceneModel(BaseSceneModel)
    ├── # The scene model the interactive display drives when the scene is one point cloud file rather than a dataset.
    ├── SUPPORTED_EXTENSIONS  # ('.ply', '.pcd', '.las', '.laz', '.off') — the extensions this scene model opens, each naming its own columns
    ├── def _load_model(self) -> PointCloud [override]
    │   ├── # Opens the file the scene path resolved to, which needs no meta data because parse_scene_path admitted only extensions this model supports.
    │   ├── calls load_point_cloud(self.resolved_path, device=self.device)
    │   └── return  # the point cloud it loaded
    ├── def extract_positions(self) -> torch.Tensor [override]
    │   ├── # Hands the display the coordinates it draws.
    │   ├── impls point_cloud = self.model
    │   ├── assert point_cloud is a PointCloud
    │   └── return  # the coordinates of point_cloud
    └── @staticmethod def parse_scene_path(path: str) -> str [override]
        ├── # Validates a direct point cloud filepath and resolves it, refusing a format whose columns only a caller could name.
        ├── assert path names an existing file
        ├── impls file_ext = the lowercased extension of path
        ├── assert file_ext sits in SUPPORTED_EXTENSIONS  # a .pth or .txt names none of its columns, and a scene model has no dataset behind it to state a layout on its behalf
        └── return  # the absolute path, by os.path.abspath
```
