# COLMAP Code Structure

## Code structure trees

`data/structures/three_d/colmap/convert.py`

```text
convert.py
├── import os
├── from concurrent.futures import ThreadPoolExecutor
├── from pathlib import Path
├── from typing import Any, Dict, List, Tuple, Union
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.extrinsics.rotation.quaternion import qvec2rotmat
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics, build_camera_intrinsics
├── from data.structures.three_d.colmap.load import ColmapCamera, ColmapImage
├── from data.structures.three_d.nerfstudio.nerfstudio_data import NerfStudio_Data
├── DEFAULT_APPLIED_TRANSFORM  # a 3x4 np.float32 array sending (x, y, z) to (x, z, -y), the transform NerfStudio records as already applied to the poses it ships
├── def convert_colmap_to_nerfstudio(filename: str, colmap_cameras: Dict[int, ColmapCamera], colmap_images: Dict[int, ColmapImage], colmap_points: Dict[int, Any], output_dir: str, ply_filename: str = "sparse_pc.ply", pixel_error_filter: float = 1.0, point_track_filter: int = 5) -> Tuple[str, str]
│   ├── # Rewrites one COLMAP reconstruction as a NerfStudio capture: a transforms record saved beside the sparse point cloud ply.
│   ├── impls output_dir made by os.makedirs, tolerating one that already exists
│   ├── impls output_path = os.path.join of output_dir under filename
│   ├── calls create_ply_from_colmap(filename=ply_filename, colmap_points=colmap_points, output_dir=output_dir, pixel_error_filter=pixel_error_filter, point_track_filter=point_track_filter)
│   ├── impls ply_path = the path it wrote that sparse cloud to
│   ├── calls _extract_intrinsics_from_colmap(colmap_cameras=colmap_cameras)
│   ├── impls intrinsic_params = the intrinsic set it read off the model's one camera
│   ├── calls _extract_cameras_from_colmap(colmap_images=colmap_images, intrinsic_params=intrinsic_params)
│   ├── impls cameras = the posed camera set it built
│   ├── calls _determine_modalities(cameras=cameras, output_dir=Path(output_dir))
│   ├── impls modalities = the modalities it found beside the images
│   ├── impls nerfstudio_intrinsic_params = the fl_x, fl_y, cx, cy, k1, k2, p1, p2 entries of intrinsic_params
│   ├── impls resolution = the h, w pair of intrinsic_params
│   ├── impls camera_model = the camera_model entry of intrinsic_params
│   ├── impls camera_intrinsics = the intrinsics of the first camera
│   ├── impls intrinsics = a 3x3 float32 torch tensor holding the fx, fy, cx, cy of camera_intrinsics, on that camera's device
│   ├── impls payload: Dict[str, Any] = an empty dict
│   ├── calls NerfStudio_Data(data=payload, device=cameras[0].device, intrinsic_params=nerfstudio_intrinsic_params, resolution=resolution, camera_model=camera_model, intrinsics=intrinsics, applied_transform=DEFAULT_APPLIED_TRANSFORM, ply_file_path=ply_filename, cameras=cameras, modalities=modalities, train_filenames=None, val_filenames=None, test_filenames=None)
│   ├── impls nerfstudio_data = the capture record it built
│   ├── calls nerfstudio_data.save(output_path=output_path)
│   └── return  # output_path, the transforms file just saved, beside ply_path, the sparse cloud written for it
├── def create_ply_from_colmap(filename: str, colmap_points: Dict[int, Any], output_dir: str, pixel_error_filter: float = 1.0, point_track_filter: int = 5) -> str
│   ├── # Writes the COLMAP sparse points surviving the reprojection-error / track-length filters as one ascii ply.
│   ├── impls output_dir made by os.makedirs, tolerating one that already exists
│   ├── impls out_path = os.path.join of output_dir under filename
│   ├── if colmap_points is empty
│   │   ├── with out_path opened for utf-8 writing as f
│   │   │   └── impls the ascii ply header written for a zero-vertex element
│   │   └── return out_path
│   ├── impls point_ids = the ids of colmap_points, sorted
│   ├── impls points = a float32 np array of the xyz of each of those points
│   ├── impls colors = a uint8 np array of the rgb of each of those points
│   ├── impls errors = a float32 np array of the reprojection error of each of those points
│   ├── impls track_lengths = a uint8 np array of how many images see each of those points
│   ├── impls valid_mask = the points whose error falls under pixel_error_filter, whose track length reaches point_track_filter
│   ├── impls num_valid_points = how many entries valid_mask keeps
│   ├── impls valid_indices = the flat indices valid_mask keeps
│   ├── with out_path opened for utf-8 writing as f
│   │   ├── impls the ascii ply header written for num_valid_points vertices
│   │   ├── def _format_point(idx: int) -> str [local]
│   │   │   ├── # Renders one point as the ply vertex line that stands for it.
│   │   │   ├── impls coord = points[idx]
│   │   │   ├── impls color = colors[idx]
│   │   │   ├── impls x, y, z = the three entries of coord
│   │   │   ├── impls r, g, b = the three entries of color
│   │   │   ├── impls vertex_line = the point's x, y, z at eight decimals before its r, g, b as integers, space-separated, newline-closed
│   │   │   └── return vertex_line
│   │   ├── impls max_workers = 32 or the number of valid indices, whichever is smaller, falling back to 1 for none
│   │   ├── with ThreadPoolExecutor(max_workers=max_workers) as executor
│   │   │   └── impls lines: List[str] = _format_point mapped by executor over valid_indices
│   │   └── impls those lines written to f
│   └── return out_path
├── def _extract_intrinsics_from_colmap(colmap_cameras: Dict[int, ColmapCamera]) -> Dict[str, Any]
│   ├── # Reads the one shared intrinsic set a COLMAP model records, restated in the undistorted OPENCV form NerfStudio writes.
│   ├── impls camera = the one camera of colmap_cameras
│   ├── impls params = camera.params
│   ├── if camera.model == "SIMPLE_PINHOLE"
│   │   ├── assert params holds 3 entries
│   │   ├── impls fl_x, fl_y = params[0] for both axes
│   │   └── impls cx, cy = params[1], params[2]
│   ├── elif camera.model == "PINHOLE"
│   │   ├── assert params holds 4 entries
│   │   ├── impls fl_x, fl_y = params[0], params[1]
│   │   └── impls cx, cy = params[2], params[3]
│   ├── elif camera.model == "OPENCV"
│   │   ├── assert params holds 8 entries
│   │   ├── assert float(params[4]) == 0.0  # f"k1 must be 0, got {params[4]}"
│   │   ├── assert float(params[5]) == 0.0  # f"k2 must be 0, got {params[5]}"
│   │   ├── assert float(params[6]) == 0.0  # f"p1 must be 0, got {params[6]}"
│   │   ├── assert float(params[7]) == 0.0  # f"p2 must be 0, got {params[7]}"
│   │   ├── impls fl_x, fl_y = params[0], params[1]
│   │   └── impls cx, cy = params[2], params[3]
│   ├── else
│   │   └── assert False  # camera.model is none of the three
│   ├── impls width, height = camera.width, camera.height
│   ├── impls intrinsic_params: Dict[str, Any] = the w, h, fl_x, fl_y, cx, cy just read, its k1, k2, p1, p2 all zero, its camera_model "OPENCV"
│   └── return intrinsic_params
├── def _extract_cameras_from_colmap(colmap_images: Dict[int, ColmapImage], intrinsic_params: Dict[str, Any]) -> Cameras
│   ├── # Poses one Cameras out of the COLMAP images, every frame carrying the model's single shared intrinsic set.
│   ├── impls intrinsics_params: Dict[str, Union[int, float]] = the fx, fy, cx, cy, h, w of intrinsic_params under the names build_camera_intrinsics takes
│   ├── impls intrinsics_list: List[CameraIntrinsics] = an empty list
│   ├── impls extrinsics_list: List[CameraExtrinsics] = an empty list
│   ├── impls camera_names: List[str] = an empty list
│   ├── impls camera_ids: List[int] = an empty list
│   ├── for each image_id, image in sorted(colmap_images.items())
│   │   ├── calls qvec2rotmat(image.qvec)
│   │   ├── impls rotation = the world-to-camera rotation that quaternion names
│   │   ├── impls translation = image.tvec reshaped to a 3x1 column
│   │   ├── impls world_to_camera = rotation beside translation, closed into 4x4 by a [0, 0, 0, 1] row
│   │   ├── impls camera_to_world = the numpy inverse of world_to_camera
│   │   ├── impls extrinsics_opencv = camera_to_world as a float32 torch tensor
│   │   ├── calls build_camera_intrinsics(model="pinhole", params=intrinsics_params, intr_convention="standard", device=extrinsics_opencv.device)
│   │   ├── impls intrinsics_list gains the intrinsics it built
│   │   ├── calls CameraExtrinsics(extrinsics=extrinsics_opencv, extr_convention="opencv", device=extrinsics_opencv.device)
│   │   ├── impls extrinsics_list gains the extrinsics it built
│   │   ├── impls camera_names gains Path(image.name).stem
│   │   └── impls camera_ids gains image_id
│   ├── assert extrinsics_list is non-empty
│   ├── calls Cameras(intrinsics=intrinsics_list, extrinsics=extrinsics_list, names=camera_names, ids=camera_ids, device=extrinsics_list[0].device)
│   ├── impls cameras = the camera set it built
│   ├── calls cameras.to(extr_convention="opengl")
│   └── return  # those cameras carried into the opengl extrinsics convention
└── def _determine_modalities(cameras: Cameras, output_dir: Path) -> List[str]
    ├── # Names which of the image, depth, normal, mask modalities output_dir can serve for every camera.
    ├── impls camera_names = the names of cameras as a list
    ├── impls modalities = a list holding "image"
    ├── impls depths_dir = the depths subdirectory of output_dir
    ├── if depths_dir is a directory
    │   ├── impls depth_names = the stems of its .npy files
    │   └── if every camera name is among depth_names
    │       └── impls modalities gains "depth"
    ├── impls normals_dir = the normals subdirectory of output_dir
    ├── if normals_dir is a directory
    │   ├── impls normal_names = the stems of its .png files
    │   └── if every camera name is among normal_names
    │       └── impls modalities gains "normal"
    ├── impls masks_dir = the masks subdirectory of output_dir
    ├── if masks_dir is a directory
    │   ├── impls mask_names = the stems of its .png files
    │   └── if every camera name is among mask_names
    │       └── impls modalities gains "mask"
    └── return modalities
```
