"""Source registry helpers for ThreeD_Scene_Dataset scene models."""

from typing import List, Type

from data.datasets.three_d_scene_registry import register_scene_models
from models.three_d.base import BaseSceneModel
from models.three_d.gso_meshes import GSOMeshesSceneModel
from models.three_d.gspl import GSPLSceneModel
from models.three_d.lapis_gs import LapisGSSceneModel
from models.three_d.letsgo import LetsGoSceneModel
from models.three_d.nerfstudio import NerfstudioSceneModel
from models.three_d.octree_gs import OctreeGSSceneModel
from models.three_d.original_3dgs import Original3DGSSceneModel
from models.three_d.point_cloud import PointCloudSceneModel
from models.three_d.two_dgs import TwoDGSSceneModel

DEFAULT_SCENE_MODELS: List[Type[BaseSceneModel]] = [
    NerfstudioSceneModel,
    OctreeGSSceneModel,
    LapisGSSceneModel,
    Original3DGSSceneModel,
    LetsGoSceneModel,
    GSPLSceneModel,
    GSOMeshesSceneModel,
    PointCloudSceneModel,
    TwoDGSSceneModel,
]

register_scene_models(scene_model_classes=DEFAULT_SCENE_MODELS)
