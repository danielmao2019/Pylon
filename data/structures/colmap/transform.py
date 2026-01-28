from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, Tuple

import numpy as np

from utils.three_d.rotation.quaternion import qvec2rotmat, rotmat2qvec


def transform_colmap(
    cameras: Dict[int, Any],
    images: Dict[int, Any],
    points3D: Dict[int, Any],
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
    # Input validations
    assert isinstance(cameras, dict), f"{type(cameras)=}"
    assert isinstance(images, dict), f"{type(images)=}"
    assert isinstance(points3D, dict), f"{type(points3D)=}"
    assert isinstance(scale, (int, float)), f"{type(scale)=}"
    assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
    assert rotation.shape == (3, 3), f"{rotation.shape=}"
    assert rotation.dtype == np.float32, f"{rotation.dtype=}"
    assert isinstance(translation, np.ndarray), f"{type(translation)=}"
    assert translation.shape == (3,), f"{translation.shape=}"
    assert translation.dtype == np.float32, f"{translation.dtype=}"

    aligned_images = transform_colmap_cameras(
        images=images,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    aligned_points = transform_colmap_points(
        points=points3D,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    return cameras, aligned_images, aligned_points


def transform_colmap_cameras(
    images: Dict[int, Any],
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Dict[int, Any]:
    # Input validations
    assert isinstance(images, dict), f"{type(images)=}"
    assert isinstance(scale, (int, float)), f"{type(scale)=}"
    assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
    assert rotation.shape == (3, 3), f"{rotation.shape=}"
    assert rotation.dtype == np.float32, f"{rotation.dtype=}"
    assert isinstance(translation, np.ndarray), f"{type(translation)=}"
    assert translation.shape == (3,), f"{translation.shape=}"
    assert translation.dtype == np.float32, f"{translation.dtype=}"

    aligned_images: Dict[int, Any] = {}
    image_ids = sorted(images)
    if len(image_ids) == 0:
        return aligned_images
    max_workers = min(32, len(image_ids))
    serializer = partial(
        _transform_colmap_camera_entry,
        images=images,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(serializer, image_ids)
        for image_id, image in results:
            aligned_images[image_id] = image
    return aligned_images


def transform_colmap_points(
    points: Dict[int, Any],
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Dict[int, Any]:
    # Input validations
    assert isinstance(points, dict), f"{type(points)=}"
    assert isinstance(scale, (int, float)), f"{type(scale)=}"
    assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
    assert rotation.shape == (3, 3), f"{rotation.shape=}"
    assert rotation.dtype == np.float32, f"{rotation.dtype=}"
    assert isinstance(translation, np.ndarray), f"{type(translation)=}"
    assert translation.shape == (3,), f"{translation.shape=}"
    assert translation.dtype == np.float32, f"{translation.dtype=}"

    aligned_points: Dict[int, Any] = {}
    point_ids = sorted(points)
    if len(point_ids) == 0:
        return aligned_points
    max_workers = min(32, len(point_ids))
    serializer = partial(
        _transform_colmap_point_entry,
        points=points,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(serializer, point_ids)
        for point_id, point in results:
            aligned_points[point_id] = point
    return aligned_points


def _transform_colmap_camera_entry(
    image_id: int,
    images: Dict[int, Any],
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Tuple[int, Any]:
    # Input validations
    assert isinstance(image_id, int), f"{type(image_id)=}"
    assert isinstance(images, dict), f"{type(images)=}"
    assert isinstance(scale, (int, float)), f"{type(scale)=}"
    assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
    assert rotation.shape == (3, 3), f"{rotation.shape=}"
    assert rotation.dtype == np.float32, f"{rotation.dtype=}"
    assert isinstance(translation, np.ndarray), f"{type(translation)=}"
    assert translation.shape == (3,), f"{translation.shape=}"
    assert translation.dtype == np.float32, f"{translation.dtype=}"

    image = images[image_id]
    assert image.id == image_id, f"{image_id=} {image.id=}"
    assert isinstance(image.qvec, np.ndarray), f"{type(image.qvec)=}"
    assert image.qvec.shape == (4,), f"{image.qvec.shape=}"
    assert isinstance(image.tvec, np.ndarray), f"{type(image.tvec)=}"
    assert image.tvec.shape == (3,), f"{image.tvec.shape=}"

    R_w2c = qvec2rotmat(image.qvec)
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ image.tvec
    R_c2w_new = rotation @ R_c2w
    t_c2w_new = scale * (rotation @ t_c2w) + translation
    R_w2c_new = R_c2w_new.T
    t_w2c_new = -R_w2c_new @ t_c2w_new
    qvec_new = rotmat2qvec(R_w2c_new)

    return (
        image_id,
        type(image)(
            id=image.id,
            qvec=qvec_new,
            tvec=t_w2c_new,
            camera_id=image.camera_id,
            name=image.name,
            xys=image.xys,
            point3D_ids=image.point3D_ids,
        ),
    )


def _transform_colmap_point_entry(
    point_id: int,
    points: Dict[int, Any],
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Tuple[int, Any]:
    # Input validations
    assert isinstance(point_id, int), f"{type(point_id)=}"
    assert isinstance(points, dict), f"{type(points)=}"
    assert isinstance(scale, (int, float)), f"{type(scale)=}"
    assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
    assert rotation.shape == (3, 3), f"{rotation.shape=}"
    assert rotation.dtype == np.float32, f"{rotation.dtype=}"
    assert isinstance(translation, np.ndarray), f"{type(translation)=}"
    assert translation.shape == (3,), f"{translation.shape=}"
    assert translation.dtype == np.float32, f"{translation.dtype=}"

    point = points[point_id]
    assert point.id == point_id, f"{point_id=} {point.id=}"
    assert isinstance(point.xyz, np.ndarray), f"{type(point.xyz)=}"
    assert point.xyz.shape == (3,), f"{point.xyz.shape=}"

    xyz_new = scale * (rotation @ point.xyz) + translation
    return (
        point_id,
        type(point)(
            id=point.id,
            xyz=xyz_new,
            rgb=point.rgb,
            error=point.error,
            image_ids=point.image_ids,
            point2D_idxs=point.point2D_idxs,
        ),
    )
