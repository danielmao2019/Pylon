# NerfStudio Code Structure

## Code structure trees

`data/structures/three_d/nerfstudio/convert.py`

```text
convert.py
├── from pathlib import Path
├── from typing import Dict
├── import numpy as np
├── from data.structures.three_d.colmap.colmap_data import COLMAP_Data
├── from data.structures.three_d.colmap.load import ColmapPoint3D
├── from data.structures.three_d.nerfstudio.nerfstudio_data import NerfStudio_Data
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── def convert_nerfstudio_to_colmap(transforms: NerfStudio_Data, point_cloud_path: str | Path) -> COLMAP_Data
│   ├── # Rewrites one NerfStudio capture as the COLMAP record of the same scene.
│   ├── impls colmap_cameras = the camera record the sibling builder makes
│   ├── impls colmap_images = the image record the sibling builder makes from it
│   ├── calls _build_colmap_points(point_cloud_path=point_cloud_path)
│   ├── impls points3d = the point record it built
│   ├── calls COLMAP_Data(cameras=colmap_cameras, images=colmap_images, points3D=points3d)
│   └── return  # the record it built
└── def _build_colmap_points(point_cloud_path: Path) -> Dict[int, ColmapPoint3D]
    ├── # Builds one COLMAP point per point of the cloud a NerfStudio capture ships beside its frames.
    ├── impls meta_data = {'xyz': {'dtype': 'float32'}}  # COLMAP records a point's coordinates as float32, and a load casts only losslessly, so stating the width refuses a capture that is not already at it
    ├── calls load_point_cloud(filepath=str(point_cloud_path), meta_data=meta_data, device='cpu')
    ├── impls pc = the PointCloud it loaded
    ├── assert pc carries an rgb field  # a COLMAP point has a color, and a cloud without one cannot make the record
    ├── impls positions = the coordinates of pc as a numpy array
    ├── impls colors = the rgb field of pc as a numpy array
    ├── impls points = an empty dict
    ├── for each point_id in range(pc.num_points)
    │   ├── calls ColmapPoint3D(id=point_id, xyz=that row's coordinate as np.float32, rgb=that row's color as np.uint8, error=0.0, image_ids=an empty int32 array, point2D_idxs=an empty int32 array)
    │   └── impls points[point_id] = the point record it built
    └── return points
```
