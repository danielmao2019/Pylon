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

    transformed_cameras = transform_nerfstudio_cameras(
        cameras=cameras,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    return transformed_cameras


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

    # Input normalizations
    rotation_tensor = torch.from_numpy(rotation).to(
        device=cameras.device,
        dtype=cameras.extrinsics.dtype,
    )
    translation_tensor = torch.from_numpy(translation).to(
        device=cameras.device,
        dtype=cameras.extrinsics.dtype,
    )

    validate_rotation_matrix(rotation)

    extrinsics = cameras.extrinsics
    rotation_c2w = extrinsics[:, :3, :3]
    translation_c2w = extrinsics[:, :3, 3]
    rotation_c2w_new = rotation_tensor @ rotation_c2w
    translation_c2w_new = scale * (translation_c2w @ rotation_tensor.T) + (
        translation_tensor
    )

    batch_size = extrinsics.shape[0]
    updated_extrinsics = (
        torch.eye(
            4,
            dtype=extrinsics.dtype,
            device=extrinsics.device,
        )
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )
    updated_extrinsics[:, :3, :3] = rotation_c2w_new
    updated_extrinsics[:, :3, 3] = translation_c2w_new

    return Cameras(
        intrinsics=cameras.intrinsics,
        extrinsics=updated_extrinsics,
        conventions=list(cameras.conventions),
        names=list(cameras.names),
        ids=list(cameras.ids),
        device=cameras.device,
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
