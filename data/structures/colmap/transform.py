from pathlib import Path
from typing import Any, Dict

import numpy as np

from data.structures.colmap.colmap import COLMAP_Data


def transform_colmap(
    colmap: Any,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Any:
    # Input validations
    assert isinstance(colmap, COLMAP_Data), f"{type(colmap)=}"
    assert isinstance(scale, (int, float)), f"{type(scale)=}"
    assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
    assert rotation.shape == (3, 3), f"{rotation.shape=}"
    assert rotation.dtype == np.float32, f"{rotation.dtype=}"
    assert isinstance(translation, np.ndarray), f"{type(translation)=}"
    assert translation.shape == (3,), f"{translation.shape=}"
    assert translation.dtype == np.float32, f"{translation.dtype=}"

    colmap.transform(
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    return colmap


def transform_colmap_binary(
    model_dir: str | Path,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Any:
    # Input validations
    assert isinstance(model_dir, (str, Path)), f"{type(model_dir)=}"
    assert isinstance(scale, (int, float)), f"{type(scale)=}"
    assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
    assert rotation.shape == (3, 3), f"{rotation.shape=}"
    assert rotation.dtype == np.float32, f"{rotation.dtype=}"
    assert isinstance(translation, np.ndarray), f"{type(translation)=}"
    assert translation.shape == (3,), f"{translation.shape=}"
    assert translation.dtype == np.float32, f"{translation.dtype=}"

    colmap = COLMAP_Data(model_dir=model_dir, file_format="binary")
    colmap.transform(
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    return colmap


def transform_colmap_text(
    model_dir: str | Path,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Any:
    # Input validations
    assert isinstance(model_dir, (str, Path)), f"{type(model_dir)=}"
    assert isinstance(scale, (int, float)), f"{type(scale)=}"
    assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
    assert rotation.shape == (3, 3), f"{rotation.shape=}"
    assert rotation.dtype == np.float32, f"{rotation.dtype=}"
    assert isinstance(translation, np.ndarray), f"{type(translation)=}"
    assert translation.shape == (3,), f"{translation.shape=}"
    assert translation.dtype == np.float32, f"{translation.dtype=}"

    colmap = COLMAP_Data(model_dir=model_dir, file_format="text")
    colmap.transform(
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    return colmap


def transform_colmap_cameras(
    images: Dict[int, Any],
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Dict[int, Any]:
    return COLMAP_Data._transform_cameras(
        images=images,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )


def transform_colmap_points(
    points: Dict[int, Any],
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Dict[int, Any]:
    return COLMAP_Data._transform_points(
        points=points,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
