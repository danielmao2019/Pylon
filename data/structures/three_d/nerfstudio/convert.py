"""Utilities for converting NerfStudio data into COLMAP data."""

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from data.structures.three_d.camera.rotation.quaternion import rotmat2qvec
from data.structures.three_d.colmap.colmap_data import COLMAP_Data
from data.structures.three_d.colmap.load import (
    ColmapCamera,
    ColmapImage,
    ColmapPoint3D,
)
from data.structures.three_d.nerfstudio.nerfstudio_data import NerfStudio_Data
from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def convert_nerfstudio_to_colmap(
    transforms: NerfStudio_Data,
    point_cloud_path: str | Path,
) -> COLMAP_Data:
    # Input validations
    assert transforms.__class__.__name__ == "NerfStudio_Data", f"{type(transforms)=}"
    assert (
        transforms.__class__.__module__
        == "data.structures.three_d.nerfstudio.nerfstudio_data"
    ), f"{type(transforms)=}"
    assert isinstance(point_cloud_path, (str, Path)), f"{type(point_cloud_path)=}"

    # Input normalizations
    point_cloud_path = Path(point_cloud_path)

    colmap_cameras = _build_colmap_cameras(transforms=transforms)
    colmap_images = _build_colmap_images(
        transforms=transforms,
        colmap_cameras=colmap_cameras,
    )
    points3d = _build_colmap_points(point_cloud_path=point_cloud_path)
    return COLMAP_Data(
        cameras=colmap_cameras,
        images=colmap_images,
        points3D=points3d,
    )


def _build_colmap_cameras(transforms: NerfStudio_Data) -> Dict[int, ColmapCamera]:
    # Input validations
    assert transforms.__class__.__name__ == "NerfStudio_Data", f"{type(transforms)=}"
    assert (
        transforms.__class__.__module__
        == "data.structures.three_d.nerfstudio.nerfstudio_data"
    ), f"{type(transforms)=}"

    width = transforms.resolution[1]
    height = transforms.resolution[0]
    focal_x = transforms.intrinsic_params["fl_x"]
    focal_y = transforms.intrinsic_params["fl_y"]
    center_x = transforms.intrinsic_params["cx"]
    center_y = transforms.intrinsic_params["cy"]
    camera_model = transforms.camera_model
    assert (
        width > 0 and height > 0
    ), f"Camera dimensions must be positive, got width={width} height={height}"
    assert (
        camera_model == "OPENCV"
    ), f"Unsupported camera model for COLMAP export: {camera_model}"
    params = np.array(
        [
            focal_x,
            focal_y,
            center_x,
            center_y,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    colmap_camera = ColmapCamera(
        id=1,
        model="OPENCV",
        width=width,
        height=height,
        params=params,
    )
    return {colmap_camera.id: colmap_camera}


def _build_colmap_images(
    transforms: NerfStudio_Data,
    colmap_cameras: Dict[int, ColmapCamera],
) -> Dict[int, ColmapImage]:
    # Input validations
    assert transforms.__class__.__name__ == "NerfStudio_Data", f"{type(transforms)=}"
    assert (
        transforms.__class__.__module__
        == "data.structures.three_d.nerfstudio.nerfstudio_data"
    ), f"{type(transforms)=}"
    assert isinstance(colmap_cameras, dict), f"{type(colmap_cameras)=}"
    assert (
        len(colmap_cameras) == 1
    ), f"Expected exactly one COLMAP camera, got {len(colmap_cameras)}"

    colmap_camera = next(iter(colmap_cameras.values()))
    images: Dict[int, ColmapImage] = {}
    for image_id, (filename, camera) in enumerate(
        zip(transforms.filenames, transforms.cameras, strict=True),
        start=1,
    ):
        assert camera.name is not None, "Camera name required for COLMAP export"
        assert camera.name == filename, f"{camera.name=} {filename=}"
        camera_opencv = camera.to(
            device=torch.device("cpu"),
            convention="opencv",
        )
        world_to_camera = camera_opencv.w2c.cpu().numpy()
        rotation = world_to_camera[:3, :3]
        translation = world_to_camera[:3, 3]
        qvec = rotmat2qvec(rotation)
        images[image_id] = ColmapImage(
            id=image_id,
            qvec=qvec,
            tvec=translation,
            camera_id=colmap_camera.id,
            name=f"{filename}.png",
            xys=np.zeros((0, 2), dtype=np.float32),
            point3D_ids=np.zeros((0,), dtype=np.int64),
        )
    return images


def _build_colmap_points(
    point_cloud_path: Path,
) -> Dict[int, ColmapPoint3D]:
    # Input validations
    assert isinstance(point_cloud_path, Path), f"{type(point_cloud_path)=}"
    assert point_cloud_path.is_file(), f"Point cloud file not found: {point_cloud_path}"

    pc: PointCloud = load_point_cloud(
        filepath=str(point_cloud_path),
        device="cpu",
        dtype=torch.float32,
    )
    assert hasattr(pc, "rgb"), "Point cloud missing RGB data for COLMAP export"
    positions = pc.xyz.cpu().numpy()
    colors = pc.rgb.cpu().numpy()
    points: Dict[int, ColmapPoint3D] = {}
    for point_id in range(pc.num_points):
        coord = positions[point_id]
        color = colors[point_id]
        points[point_id] = ColmapPoint3D(
            id=point_id,
            xyz=np.asarray(coord, dtype=np.float32),
            rgb=np.asarray(color, dtype=np.uint8),
            error=0.0,
            image_ids=np.zeros((0,), dtype=np.int32),
            point2D_idxs=np.zeros((0,), dtype=np.int32),
        )
    return points
