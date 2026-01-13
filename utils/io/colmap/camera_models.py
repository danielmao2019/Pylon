"""COLMAP camera model definitions."""

import collections
from typing import Dict

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])


# COLMAP camera models
CAMERA_MODELS: Dict[int, CameraModel] = {
    0: CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    1: CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    2: CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    3: CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    4: CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    5: CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    6: CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
}

CAMERA_MODEL_NAME_TO_ID: Dict[str, int] = {
    model.model_name: model_id for model_id, model in CAMERA_MODELS.items()
}
