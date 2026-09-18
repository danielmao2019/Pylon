# COLMAP Code Structure

## Code structure trees

`data/structures/three_d/colmap/convert.py`

```text
convert.py
├── import os
├── from concurrent.futures import ThreadPoolExecutor
├── from pathlib import Path
├── from typing import Any, Dict, List, Tuple
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.extrinsics.rotation.quaternion import quat_to_rotmat
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
│   ├── def _colmap_to_pylon [local]
│   │   ├── # Read the COLMAP cameras and images as the one posed Cameras Pylon holds them in, every image sharing the model's one intrinsics.
│   │   ├── impls sorted_images = the (image_id, image) pairs of colmap_images, sorted by image_id
│   │   ├── impls camera_ids: List[int] = an empty list
│   │   ├── impls camera_names: List[str] = an empty list
│   │   ├── impls images: List[ColmapImage] = an empty list
│   │   ├── for each image_id, image of sorted_images
│   │   │   ├── impls append image_id to camera_ids
│   │   │   ├── impls append the stem of image.name as a Path to camera_names
│   │   │   └── impls append image to images
│   │   ├── calls _extract_intrinsics_from_colmap(colmap_cameras=colmap_cameras)
│   │   ├── impls intrinsics = the unbatched CameraIntrinsics it built  # broadcast over every image, since one COLMAP camera governs them all
│   │   ├── calls _extract_extrinsics_from_colmap(images=images)
│   │   ├── impls extrinsics = the batched CameraExtrinsics it built
│   │   ├── calls Cameras(intrinsics=intrinsics, extrinsics=extrinsics, names=camera_names, ids=camera_ids, device=extrinsics.device)
│   │   ├── impls cameras = the camera set it built
│   │   ├── calls cameras.to(extr_convention="opengl")
│   │   └── return  # those cameras carried into the opengl extrinsics convention
│   ├── calls _colmap_to_pylon
│   ├── impls cameras = the posed camera set it built
│   ├── def _pylon_to_nerfstudio [local]
│   │   ├── # Write the posed Cameras as the NerfStudio transforms record at output_path, the ply beside it.
│   │   ├── calls _determine_modalities(cameras=cameras, output_dir=Path(output_dir))
│   │   ├── impls modalities = the modalities it found beside the images
│   │   ├── impls camera_intrinsics = cameras.intrinsics  # the batch's one intrinsics, no camera of it being the one read
│   │   ├── impls capture_params = {key: the one value camera_intrinsics.params[key] holds for each key of camera_intrinsics.params}  # one, since the unbatched intrinsics is shared by every image
│   │   ├── impls nerfstudio_intrinsic_params = the fl_x, fl_y, cx, cy of capture_params, its k1, k2, p1, p2 all zero  # the undistorted OPENCV form NerfStudio writes
│   │   ├── impls resolution = the h, w pair of capture_params
│   │   ├── impls camera_model = "OPENCV"
│   │   ├── impls intrinsics = a 3x3 float32 torch tensor holding the fx, fy, cx, cy of capture_params, on cameras.device
│   │   ├── impls payload: Dict[str, Any] = an empty dict
│   │   ├── calls NerfStudio_Data(data=payload, device=cameras.device, intrinsic_params=nerfstudio_intrinsic_params, resolution=resolution, camera_model=camera_model, intrinsics=intrinsics, applied_transform=DEFAULT_APPLIED_TRANSFORM, ply_file_path=ply_filename, cameras=cameras, modalities=modalities, train_filenames=None, val_filenames=None, test_filenames=None)
│   │   ├── impls nerfstudio_data = the capture record it built
│   │   ├── calls nerfstudio_data.save(output_path=output_path)
│   │   └── return
│   ├── calls _pylon_to_nerfstudio
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
├── def _extract_intrinsics_from_colmap(colmap_cameras: Dict[int, ColmapCamera]) -> CameraIntrinsics
│   ├── # Reads the one pinhole a COLMAP model records as the unbatched CameraIntrinsics every image shares.
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
│   ├── calls build_camera_intrinsics(model="pinhole", params={"fx": fl_x, "fy": fl_y, "cx": cx, "cy": cy, "h": height, "w": width}, intr_convention="standard")
│   └── return  # the unbatched CameraIntrinsics it built
├── def _extract_extrinsics_from_colmap(images: List[ColmapImage]) -> CameraExtrinsics
│   ├── # Poses the COLMAP images, in the order given, as one batched opencv CameraExtrinsics.
│   ├── impls qvecs, tvecs = two empty lists
│   ├── for each image of images
│   │   ├── impls append image.qvec to qvecs
│   │   └── impls append image.tvec to tvecs
│   ├── impls quaternions = the [N, 4] np stack of qvecs  # the whole batch's pose stack is built in one op
│   ├── impls translations = the [N, 3] np stack of tvecs
│   ├── assert quaternions is floating  # both stacks are cast to float64 next
│   ├── assert translations is floating
│   ├── calls quat_to_rotmat(quaternions=quaternions as a float64 torch tensor)
│   ├── impls rotation = the [N, 3, 3] world-to-camera rotations it returned, transposed  # COLMAP states the pose world-to-camera, so cam2world is its rigid inverse
│   ├── impls translation = translations as a float64 torch tensor
│   ├── impls camera_to_world = N float64 4x4 identities
│   ├── impls camera_to_world's rotation blocks = rotation
│   ├── impls camera_to_world's translation blocks = -(rotation @ translation)  # the rigid inverse's translation: the transposed rotation applied to the negated translation
│   ├── impls extrinsics_opencv = camera_to_world as a float32 torch tensor
│   ├── calls CameraExtrinsics(extrinsics=extrinsics_opencv, extr_convention="opencv", device=extrinsics_opencv.device)
│   └── return  # the batched CameraExtrinsics it built
└── def _determine_modalities(cameras: Cameras, output_dir: Path) -> List[str]
    ├── # Names which of the image, depth, normal, mask modalities output_dir can serve for every camera.
    ├── impls camera_names = the names of cameras as a list
    ├── impls modalities = a list holding "image"
    ├── impls depths_dir = the depths subdirectory of output_dir
    ├── if depths_dir is a directory
    │   ├── impls depth_names = the stems of its .npy files
    │   └── if every camera name is among depth_names
    │       └── impls append "depth" to modalities
    ├── impls normals_dir = the normals subdirectory of output_dir
    ├── if normals_dir is a directory
    │   ├── impls normal_names = the stems of its .png files
    │   └── if every camera name is among normal_names
    │       └── impls append "normal" to modalities
    ├── impls masks_dir = the masks subdirectory of output_dir
    ├── if masks_dir is a directory
    │   ├── impls mask_names = the stems of its .png files
    │   └── if every camera name is among mask_names
    │       └── impls append "mask" to modalities
    └── return modalities
```
