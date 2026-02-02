from typing import Dict

import numpy as np
import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.validation import validate_rotation_matrix
from data.structures.three_d.point_cloud.ops.apply_transform import apply_transform
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def transform_nerfstudio(
    cameras: Cameras,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Cameras:
    # Input validations
    assert isinstance(cameras, Cameras), f"{type(cameras)=}"
    assert isinstance(scale, (int, float)), f"{type(scale)=}"
    assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
    assert rotation.shape == (3, 3), f"{rotation.shape=}"
    assert rotation.dtype == np.float32, f"{rotation.dtype=}"
    assert isinstance(translation, np.ndarray), f"{type(translation)=}"
    assert translation.shape == (3,), f"{translation.shape=}"
    assert translation.dtype == np.float32, f"{translation.dtype=}"

    validate_rotation_matrix(rotation)

    return transform_nerfstudio_cameras(
        cameras=cameras,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )


def transform_nerfstudio_cameras(
    cameras: Cameras,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Cameras:
    # Input validations
    assert isinstance(cameras, Cameras), f"{type(cameras)=}"
    assert isinstance(scale, (int, float)), f"{type(scale)=}"
    assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
    assert rotation.shape == (3, 3), f"{rotation.shape=}"
    assert rotation.dtype == np.float32, f"{rotation.dtype=}"
    assert isinstance(translation, np.ndarray), f"{type(translation)=}"
    assert translation.shape == (3,), f"{translation.shape=}"
    assert translation.dtype == np.float32, f"{translation.dtype=}"

    validate_rotation_matrix(rotation)
    return cameras.transform(
        scale=scale,
        rotation=rotation,
        translation=translation,
    )


def transform_nerfstudio_points(
    point_cloud: PointCloud,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> PointCloud:
    # Input validations
    assert isinstance(point_cloud, PointCloud), f"{type(point_cloud)=}"
    assert isinstance(scale, (int, float)), f"{type(scale)=}"
    assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
    assert rotation.shape == (3, 3), f"{rotation.shape=}"
    assert rotation.dtype == np.float32, f"{rotation.dtype=}"
    assert isinstance(translation, np.ndarray), f"{type(translation)=}"
    assert translation.shape == (3,), f"{translation.shape=}"
    assert translation.dtype == np.float32, f"{translation.dtype=}"

    validate_rotation_matrix(rotation)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation * float(scale)
    transform[:3, 3] = translation

    transformed_xyz = apply_transform(points=point_cloud.xyz, transform=transform)
    assert isinstance(transformed_xyz, torch.Tensor), f"{type(transformed_xyz)=}"

    data: Dict[str, torch.Tensor] = {"xyz": transformed_xyz}
    for name in point_cloud.field_names()[1:]:
        data[name] = getattr(point_cloud, name)

    return PointCloud(data=data)
