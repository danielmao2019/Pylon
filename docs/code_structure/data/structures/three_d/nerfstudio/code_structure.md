# NerfStudio Code Structure

## Code structure trees

`data/structures/three_d/nerfstudio/convert.py`

```text
convert.py
├── from pathlib import Path
├── from typing import Dict, Union
├── import numpy as np
├── import torch
├── from data.structures.three_d.colmap.colmap_data import COLMAP_Data
├── from data.structures.three_d.colmap.load import ColmapPoint3D
├── from data.structures.three_d.nerfstudio.nerfstudio_data import NerfStudio_Data
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── def convert_nerfstudio_to_colmap(transforms: NerfStudio_Data, point_cloud_path: Union[str, Path]) -> COLMAP_Data
│   ├── # Rewrites one NerfStudio capture as the COLMAP record of the same scene.
│   ├── impls colmap_cameras = the camera record the sibling builder makes
│   ├── impls colmap_images = the image record the sibling builder makes from it
│   ├── calls _build_colmap_points(point_cloud_path=point_cloud_path)
│   ├── impls points3d = the point record it built
│   ├── calls COLMAP_Data(cameras=colmap_cameras, images=colmap_images, points3D=points3d)
│   └── return  # the record it built
└── def _build_colmap_points(point_cloud_path: Path) -> Dict[int, ColmapPoint3D]
    ├── # Builds one COLMAP point per point of the cloud a NerfStudio capture ships beside its frames.
    ├── calls load_point_cloud(filepath=str(point_cloud_path), device='cpu', dtype=torch.float32)
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

`data/structures/three_d/nerfstudio/load.py`

```text
load.py
├── import json
├── from pathlib import Path
├── from typing import Any, Dict, List, Optional, Tuple, Union
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.camera.intrinsics.validation import validate_camera_intrinsics_params
├── from data.structures.three_d.nerfstudio.validate import MODALITY_SPECS, validate_applied_transform_data, validate_camera_model_data, validate_data, validate_frames_data, validate_intrinsic_params, validate_intrinsics_data, validate_ply_file_path_data, validate_resolution_data, validate_split_filenames_data
├── def load_nerfstudio_data(filepath: Union[str, Path], device: Union[str, torch.device] = torch.device("cuda")) -> Tuple[Dict[str, Any], Dict[str, Union[float, int]], Tuple[int, int], str, torch.Tensor, np.ndarray, str, Cameras, List[str], Optional[List[str]], Optional[List[str]], Optional[List[str]]]
│   ├── # Opens a NerfStudio transforms.json, validates each section of the record it holds, and returns the record beside every section read out of it.
│   ├── assert isinstance(filepath, (str, Path))        # f"{type(filepath)=}"
│   ├── assert isinstance(device, (str, torch.device))  # f"{type(device)=}"
│   ├── impls path = Path(filepath).resolve()
│   ├── impls target_device = torch.device(device)
│   ├── assert path.is_file()  # f"transforms.json not found: {path}"
│   ├── with path.open("r", encoding="utf-8") as handle
│   │   └── impls data: Dict[str, Any] = json.load(handle)
│   ├── calls validate_data(data)
│   ├── calls validate_intrinsic_params(data)
│   ├── calls validate_resolution_data(data)
│   ├── calls validate_camera_model_data(data)
│   ├── calls validate_intrinsics_data(data)
│   ├── calls validate_applied_transform_data(data)
│   ├── calls validate_ply_file_path_data(data=data, root_dir=path.parent)
│   ├── calls validate_frames_data(data=data, root_dir=path.parent)
│   ├── calls validate_split_filenames_data(data)
│   ├── calls load_intrinsic_params(data)
│   ├── impls intrinsic_params = the entries it picked
│   ├── calls load_resolution(data)
│   ├── impls resolution = the (h, w) it read
│   ├── calls load_camera_model(data)
│   ├── impls camera_model = the model name it read
│   ├── calls load_intrinsics(data=data, device=target_device)
│   ├── impls intrinsics = the K matrix it built
│   ├── calls load_applied_transform(data)
│   ├── impls applied_transform = the array it converted
│   ├── calls load_ply_file_path(data)
│   ├── impls ply_file_path = the path it read
│   ├── calls load_split_filenames(data)
│   ├── impls train_filenames, val_filenames, test_filenames = the three it read
│   ├── calls load_cameras(data=data, device=target_device)
│   ├── impls cameras = the cameras it built
│   ├── calls load_modalities(data)
│   ├── impls modalities = the modalities it collected
│   └── return data, intrinsic_params, resolution, camera_model, intrinsics, applied_transform, ply_file_path, cameras, modalities, train_filenames, val_filenames, test_filenames
├── def load_intrinsic_params(data: Dict[str, Any]) -> Dict[str, float | int]
│   ├── # Picks the focal, principal-point and k1, k2, p1, p2 distortion entries out of a NerfStudio transforms record, keeping the record's own key names.
│   ├── impls keys = ["fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2"]
│   ├── impls build the dict mapping each key in keys to data[key]
│   └── return  # the dict it built
├── def load_resolution(data: Dict[str, Any]) -> Tuple[int, int]
│   ├── # Reads the image size a NerfStudio transforms record states, height first.
│   └── return data["h"], data["w"]
├── def load_camera_model(data: Dict[str, Any]) -> str
│   ├── # Reads the camera model name a NerfStudio transforms record states.
│   └── return data["camera_model"]
├── def load_intrinsics(data: Dict[str, Any], device: Union[str, torch.device] = torch.device("cpu")) -> torch.Tensor
│   ├── # Builds the 3x3 pinhole K matrix of a NerfStudio transforms record's fl_x, fl_y, cx and cy, then passes those and its h and w as Python scalars to the standard-frame pinhole params validation.
│   ├── impls intrinsics = the float32 [[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]] tensor of data's float-cast entries on torch.device(device)
│   ├── calls validate_camera_intrinsics_params(model="pinhole", intr_convention="standard", params=the record's fl_x, fl_y, cx, cy as float fx, fy, cx, cy, its h, w as ints)
│   └── return intrinsics
├── def load_applied_transform(data: Dict[str, Any]) -> np.ndarray
│   ├── # Reads the applied_transform a NerfStudio transforms record carries, as a float32 array.
│   ├── impls convert data["applied_transform"] to a float32 numpy array
│   └── return  # the array it converted
├── def load_ply_file_path(data: Dict[str, Any]) -> str
│   ├── # Reads the point cloud path a NerfStudio transforms record names.
│   └── return data["ply_file_path"]
├── def load_cameras(data: Dict[str, Any], device: Union[str, torch.device] = torch.device("cpu")) -> Cameras
│   ├── # Reads the frames of one NerfStudio transforms record as the cameras that posed them.
│   ├── impls frames: List[Any] = the frames the record lists
│   ├── impls intrinsics_params = the record's fl_x, fl_y, cx, cy as float fx, fy, cx, cy, its h, w as ints
│   ├── calls build_camera_intrinsics(model="pinhole", params=each intrinsics_params value broadcast to a float32 [len(frames)] tensor on device, intr_convention="standard", device=device)  # the record's one top-level pinhole governs every frame
│   ├── impls intrinsics = the batched CameraIntrinsics it built
│   ├── calls CameraExtrinsics(extrinsics=every frame's transform_matrix stacked as a float32 [len(frames), 4, 4] tensor on device, extr_convention="opengl", device=device)
│   ├── impls extrinsics = the batched CameraExtrinsics it built
│   ├── impls names: List[Optional[str]] = the stem of each frame's file_path as a Path
│   ├── for each frame in frames
│   │   ├── if frame carries a colmap_im_id
│   │   │   └── impls ids gains that colmap_im_id
│   │   └── else
│   │       └── impls ids gains None
│   ├── calls Cameras(intrinsics=intrinsics, extrinsics=extrinsics, names=names, ids=ids, device=device)
│   └── return  # the cameras it built
├── def load_modalities(data: Dict[str, Any]) -> List[str]
│   ├── # Names the modalities a NerfStudio record's frames carry, judged by which modality path keys (each MODALITY_SPECS spec's first entry) its first frame holds.
│   ├── impls frames: List[Any] = data["frames"]
│   ├── impls collect each modality of MODALITY_SPECS whose spec's first entry is a key of frames[0]
│   └── return  # the modalities it collected
└── def load_split_filenames(data: Dict[str, Any]) -> Tuple[List[str] | None, List[str] | None, List[str] | None]
    ├── # Reads the train, val and test filename lists of a NerfStudio transforms record, a None for each when it carries no train_filenames.
    ├── if "train_filenames" not in data
    │   └── return None, None, None
    └── return data["train_filenames"], data["val_filenames"], data["test_filenames"]
```
