"""Registry for 3D scene model classes used by ThreeD_Scene_Dataset."""

from typing import List, Type

from models.three_d.base import BaseSceneModel

_SCENE_MODEL_REGISTRY: List[Type[BaseSceneModel]] = []


def register_scene_model(scene_model_cls: Type[BaseSceneModel]) -> None:
    assert isinstance(scene_model_cls, type) and issubclass(
        scene_model_cls, BaseSceneModel
    ), f"scene_model_cls must be a BaseSceneModel subclass, got {scene_model_cls}"
    if scene_model_cls not in _SCENE_MODEL_REGISTRY:
        _SCENE_MODEL_REGISTRY.append(scene_model_cls)


def register_scene_models(scene_model_classes: List[Type[BaseSceneModel]]) -> None:
    assert isinstance(scene_model_classes, list), f"{type(scene_model_classes)=}"
    for scene_model_cls in scene_model_classes:
        register_scene_model(scene_model_cls=scene_model_cls)


def registered_scene_models() -> List[Type[BaseSceneModel]]:
    return list(_SCENE_MODEL_REGISTRY)
