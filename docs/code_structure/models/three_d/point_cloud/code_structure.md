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
    ├── def _load_model(self) -> PointCloud [override]
    │   ├── # Opens the file the scene path resolved to, onto the device the scene model was built for.
    │   ├── calls load_point_cloud(self.resolved_path, device=self.device)
    │   └── return  # the point cloud it loaded
    ├── def extract_positions(self) -> torch.Tensor [override]
    │   ├── # Hands the display the coordinates it draws.
    │   ├── impls point_cloud = self.model
    │   ├── assert point_cloud is a PointCloud
    │   └── return  # the coordinates of point_cloud
    └── @staticmethod def parse_scene_path(path: str) -> str [override]
        ├── # Validates a direct point cloud filepath and resolves it.
        ├── assert path names an existing file
        └── return  # the absolute path, by os.path.abspath
```
