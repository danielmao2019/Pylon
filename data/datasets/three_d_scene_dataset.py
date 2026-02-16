import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import dash
import torch

from data.datasets.base_dataset import BaseDataset
from data.datasets.three_d_scene_registry import (
    register_scene_model,
    register_scene_models,
    registered_scene_models,
)
from data.structures.three_d.nerfstudio import NerfStudio_Data
from models.three_d.base import BaseSceneModel


class ThreeD_Scene_Dataset(BaseDataset):
    """Dataset for loading iVISION scenes across heterogeneous modalities.

    Each entry in ``scene_paths`` may describe a Gaussian Splatting checkpoint
    (Nerfstudio, original 3DGS, Octree GS, or 2DGS), a mesh directory, or a
    point-cloud file. The dataset normalizes those paths, infers the scene type,
    and provides the glue needed for the viewer to render them side-by-side.
    """

    SPLIT_OPTIONS = []
    INPUT_NAMES = ['model']
    LABEL_NAMES = []

    def __init__(
        self,
        scene_paths: List[str],
        device: Optional[torch.device] = torch.device('cuda'),
        **kwargs,
    ) -> None:
        """Instantiate the dataset from explicit scene asset paths.

        Args:
            scene_paths: Directories or files that identify the scenes to load.
            device: Target torch device for instantiated scene models.
            **kwargs: Additional options forwarded to :class:`BaseDataset`.
        """
        assert scene_paths, "Expected a non-empty list of scene checkpoint directories"

        self.device = (
            torch.device(device) if device is not None else torch.device('cuda')
        )

        self.scene_paths: List[str] = []

        raw_scene_types: List[Type[BaseSceneModel]] = []
        for path in scene_paths:
            resolved_path, scene_type = ThreeD_Scene_Dataset.parse_scene_path(path)
            self.scene_paths.append(resolved_path)
            raw_scene_types.append(scene_type)

        super().__init__(device=self.device, **kwargs)

    @classmethod
    def register_scene_model_class(
        cls,
        scene_model_cls: Type[BaseSceneModel],
    ) -> None:
        register_scene_model(scene_model_cls=scene_model_cls)

    @classmethod
    def register_scene_model_classes(
        cls,
        scene_model_classes: List[Type[BaseSceneModel]],
    ) -> None:
        register_scene_models(scene_model_classes=scene_model_classes)

    @staticmethod
    def parse_scene_path(path: str) -> Tuple[str, Type[BaseSceneModel]]:
        """Resolve an input `path` to a concrete scene path and determine model class."""
        normalized_path = os.path.abspath(path)
        scene_model_classes = registered_scene_models()
        assert scene_model_classes, (
            "No scene model classes are registered. "
            "Call register_scene_model(...) or register_scene_models(...) before "
            "constructing ThreeD_Scene_Dataset. "
            "Project-scoped scene models can be registered via "
            "register_scene_models(scene_model_classes=project.models.three_d.registry.DEFAULT_SCENE_MODELS)."
        )
        successes: List[Tuple[str, Type[BaseSceneModel]]] = []
        failure_messages: List[str] = []
        for model_cls in scene_model_classes:
            try:
                resolved_path = model_cls.parse_scene_path(normalized_path)
            except Exception as exc:
                label = getattr(model_cls, '__name__', str(model_cls))
                failure_messages.append(f"{label}: {exc}")
                continue
            successes.append((resolved_path, model_cls))

        assert (
            successes
        ), f"Could not parse scene path '{path}'. Errors: {failure_messages}"
        assert (
            len(successes) == 1
        ), f"Ambiguous scene type for '{path}'; successes={successes}"

        return successes[0]

    def _init_annotations(self) -> None:
        """Populate ``self.annotations`` for the resolved scene list.

        For every parsed scene we cache canonical intrinsics (from
        ``transforms.json`` in the associated data directory) and record the
        metadata needed by the viewer to materialize the scene on demand.
        """
        self.annotations = []
        for scene_path in self.scene_paths:
            resolved_path, scene_type = ThreeD_Scene_Dataset.parse_scene_path(
                scene_path
            )
            scene_name = scene_type.extract_scene_name(resolved_path)
            data_dir = scene_type.infer_data_dir(resolved_path)

            assert (
                data_dir is not None
            ), f"Unable to determine data directory for scene '{scene_name}'"

            transforms_path = Path(data_dir) / 'transforms.json'
            assert (
                transforms_path.is_file()
            ), f"transforms.json not found under data directory '{data_dir}'"
            try:
                transforms = NerfStudio_Data.load(
                    filepath=transforms_path,
                    device=self.device,
                )
            except Exception as e:
                raise type(e)(f"{e}. {str(transforms_path)=}")

            self.annotations.append(
                {
                    'scene_path': scene_path,
                    'scene_name': scene_name,
                    'scene_type': scene_type,
                    'data_dir': data_dir,
                    'transforms_data': transforms,
                }
            )

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Extend cache versioning with checkpoint directories."""
        version_dict = super()._get_cache_version_dict()
        version_dict['scene_paths'] = self.scene_paths
        return version_dict

    @staticmethod
    def _extract_positions(scene_model: BaseSceneModel) -> torch.Tensor:
        """Retrieve position tensor for different scene representations."""
        points = scene_model.extract_positions()

        assert isinstance(
            points, torch.Tensor
        ), f"Expected torch.Tensor for scene positions, got {type(points)}"

        assert (
            points.ndim == 2 and points.shape[1] == 3
        ), f"Scene positions must have shape [N, 3], got {points.shape}"

        return points.detach()

    def _load_datapoint(
        self, idx: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Materialize the scene asset referenced by ``idx``.

        Args:
            idx: Position within ``self.annotations``.

        Returns:
            inputs: Dict with the loaded scene model wrapper.
            labels: Always empty because this dataset is visualization-only.
            meta_info: Annotation metadata plus basic spatial statistics.
        """
        annotation = self.annotations[idx]
        cache_handle = self.cache if getattr(self, 'cache', None) is not None else None
        cache_key: Optional[str] = None
        if cache_handle is not None:
            cache_key = self._get_cache_filepath(idx)
        scene_model = annotation['scene_type'](
            scene_path=annotation['scene_path'],
            device=self.device,
            cache=cache_handle,
            cache_key=cache_key,
        )

        inputs = {'model': scene_model}
        labels = {}
        meta_info = {
            k: (v.__name__ if k == 'scene_type' else v) for k, v in annotation.items()
        }

        positions = self._extract_positions(scene_model=scene_model)
        positions = positions.to(dtype=torch.float32)
        position_mean = positions.mean(dim=0)
        position_min = positions.min(dim=0).values
        position_max = positions.max(dim=0).values

        meta_info['position_mean'] = tuple(
            float(x) for x in position_mean.cpu().tolist()
        )
        meta_info['position_min'] = tuple(float(x) for x in position_min.cpu().tolist())
        meta_info['position_max'] = tuple(float(x) for x in position_max.cpu().tolist())

        return inputs, labels, meta_info

    def register_callbacks(
        self, app: dash.Dash, viewer: Any, dataset_name: str, scene_name: str
    ) -> None:
        """Register modality-specific callbacks on the provided Dash app.

        Currently Octree GS and LapisGS scenes contribute custom callbacks.
        """
        registry_attr = '_ivision_registered_scene_models'
        existing = getattr(app, registry_attr, None)
        registered: Set[Type[BaseSceneModel]]
        if isinstance(existing, set):
            registered = existing
        else:
            registered = set()

        for annotation in self.annotations:
            scene_type = annotation['scene_type']
            if scene_type in registered:
                continue
            assert viewer is not None, 'viewer instance required to register callbacks'
            scene_type.register_callbacks(
                dataset=self,
                app=app,
                viewer=viewer,
                dataset_name=dataset_name,
                scene_name=scene_name,
            )
            registered.add(scene_type)

        setattr(app, registry_attr, registered)

    def setup_states(
        self,
        app: dash.Dash,
        dataset_name: str,
        scene_name: str,
        method_names: List[str],
    ) -> None:
        """Prepare viewer state for each represented scene type."""
        for annotation, method_name in zip(self.annotations, method_names, strict=True):
            scene_type = annotation['scene_type']
            scene_type.setup_states(
                app=app,
                dataset_name=dataset_name,
                scene_name=scene_name,
                method_name=method_name,
            )

    @staticmethod
    def display_datapoint(
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None,
    ) -> Optional['html.Div']:
        """Hook for dataset-specific HTML rendering (unused).

        Returning ``None`` delegates the display to the viewer's default layout.
        """
        return None
