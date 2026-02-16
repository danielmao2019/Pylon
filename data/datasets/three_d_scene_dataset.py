import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import dash
import torch
from nerfstudio.pipelines.base_pipeline import Pipeline
from pytorch3d.structures import Meshes

from data.datasets.base_dataset import BaseDataset
from data.structures.three_d.nerfstudio import NerfStudio_Data
from data.structures.three_d.point_cloud import PointCloud
from models.three_d.anytime_gs import AnytimeGSSceneModel
from models.three_d.base import BaseSceneModel
from models.three_d.cheng2025clod import Cheng2025CLODSceneModel
from models.three_d.gso_meshes import GSOMeshesSceneModel
from models.three_d.gspl import GSPLSceneModel
from models.three_d.gspl.model import GSPLModel
from models.three_d.ivision_meshes import iVISIONMeshesSceneModel
from models.three_d.lapis_gs import LapisGSSceneModel
from models.three_d.letsgo import LetsGoSceneModel
from models.three_d.meshes import BaseMeshesSceneModel
from models.three_d.milef2025learning import Milef2025LearningSceneModel
from models.three_d.nerfstudio import NerfstudioSceneModel
from models.three_d.octree_gs import OctreeGSSceneModel
from models.three_d.octree_gs.loader import OctreeGS_3DGS
from models.three_d.original_3dgs import Original3DGSSceneModel
from models.three_d.original_3dgs.loader import GaussianModel as GaussianModel3D
from models.three_d.point_cloud import PointCloudSceneModel
from models.three_d.visibility_ordered_gs import VisibilityOrderedGSSceneModel

SCENE_MODEL_CLASSES: List[Type[BaseSceneModel]] = [
    NerfstudioSceneModel,
    OctreeGSSceneModel,
    LapisGSSceneModel,
    AnytimeGSSceneModel,
    Milef2025LearningSceneModel,
    Cheng2025CLODSceneModel,
    VisibilityOrderedGSSceneModel,
    Original3DGSSceneModel,
    LetsGoSceneModel,
    GSPLSceneModel,
    iVISIONMeshesSceneModel,
    GSOMeshesSceneModel,
    PointCloudSceneModel,
]


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

        # Base dataset expects canonical scene names to locate DJI_Processed, etc.
        if any(cls is OctreeGSSceneModel for cls in raw_scene_types):
            kwargs['use_disk_cache'] = False

        super().__init__(device=self.device, **kwargs)

    @staticmethod
    def parse_scene_path(path: str) -> Tuple[str, Type[BaseSceneModel]]:
        """Resolve an input `path` to a concrete scene path and determine model class."""
        normalized_path = os.path.abspath(path)
        successes: List[Tuple[str, Type[BaseSceneModel]]] = []
        failure_messages: List[str] = []
        for model_cls in SCENE_MODEL_CLASSES:
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
        model_cls = type(scene_model)

        if isinstance(scene_model, PointCloudSceneModel):
            data = scene_model.model
            assert isinstance(
                data, PointCloud
            ), "Expected PointCloud for point_cloud scene data"
            points = data.xyz
        elif isinstance(scene_model, BaseMeshesSceneModel):
            data = scene_model.model
            assert isinstance(
                data, Meshes
            ), "Expected PyTorch3D Meshes for mesh scene data"
            verts_list = data.verts_list()
            assert verts_list, "Mesh scene data contains no vertices"
            points = torch.cat(verts_list, dim=0)
        elif isinstance(scene_model, NerfstudioSceneModel):
            assert isinstance(
                scene_model.model, Pipeline
            ), "Expected nerfstudio Pipeline for 3dgs scene data"
            points = scene_model.model.model.means
        elif isinstance(scene_model, Original3DGSSceneModel):
            data = scene_model.model
            assert isinstance(
                data, GaussianModel3D
            ), "Expected GaussianModel for 3dgs_original scene data"
            points = data.get_xyz
        elif isinstance(scene_model, OctreeGSSceneModel):
            data = scene_model.model
            assert isinstance(data, OctreeGS_3DGS)
            points = data.get_anchor
        elif isinstance(scene_model, LapisGSSceneModel):
            data = scene_model.model
            assert isinstance(
                data, GaussianModel3D
            ), "Expected GaussianModel for lapis_gs scene data"
            points = data.get_xyz
        elif isinstance(scene_model, LetsGoSceneModel):
            data = scene_model.model
            assert hasattr(data, 'gaussians'), "LetsGo model missing gaussians list"
            points = torch.cat([gaussian.get_xyz for gaussian in data.gaussians], dim=0)
        elif isinstance(scene_model, GSPLSceneModel):
            data = scene_model.model
            assert isinstance(data, GSPLModel), "Expected GSPLModel for gspl scene data"
            points = data.get_xyz
        elif isinstance(scene_model, Milef2025LearningSceneModel):
            data = scene_model.model
            assert isinstance(
                data, GaussianModel3D
            ), "Expected GaussianModel for Milef2025Learning scene data"
            points = data.get_xyz
        elif isinstance(scene_model, Cheng2025CLODSceneModel):
            data = scene_model.model
            assert hasattr(data, 'get_xyz'), "Cheng2025CLOD model missing get_xyz"
            points = data.get_xyz
        elif isinstance(scene_model, AnytimeGSSceneModel):
            data = scene_model.model
            assert isinstance(
                data, GaussianModel3D
            ), "Expected GaussianModel for AnytimeGS scene data"
            points = data.get_xyz
        elif isinstance(scene_model, VisibilityOrderedGSSceneModel):
            data = scene_model.model
            assert isinstance(
                data, GaussianModel3D
            ), "Expected GaussianModel for visibility-ordered scene data"
            points = data.get_xyz
        else:
            scene_type_label = getattr(model_cls, '__name__', str(model_cls))
            raise ValueError(f"Unsupported scene type '{scene_type_label}'")

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
