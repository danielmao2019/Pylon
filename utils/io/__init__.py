"""
UTILS.IO API
"""

import importlib

from utils.io.image import load_image
from utils.io.point_clouds.load_point_cloud import load_point_cloud
from utils.io.point_clouds.save_point_cloud import save_point_cloud
from utils.io.json import serialize_tensor, save_json, load_json

_2dgs_module = importlib.import_module('utils.io.2dgs')
load_2dgs_model = _2dgs_module.load_2dgs_model


__all__ = (
    'load_image',
    'load_point_cloud',
    'save_point_cloud',
    'serialize_tensor',
    'save_json',
    'load_json',
    'load_2dgs_model',
)
