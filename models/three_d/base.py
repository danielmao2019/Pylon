import os
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dash
from dash import html
import torch
import xxhash

from data.cache import CombinedDatasetCache
from data.structures.three_d.camera.camera import Camera
from models.three_d import styles as three_d_styles
from utils.ops import apply_tensor_op
from data.structures.three_d.camera.render_camera import render_camera


class BaseSceneModel(ABC):

    def __init__(
        self,
        scene_path: str,
        scene_name: Optional[str] = None,
        data_dir: Optional[str | Path] = None,
        device: torch.device = torch.device('cuda'),
        cache: Optional[CombinedDatasetCache] = None,
        cache_key: Optional[str] = None,
    ) -> None:
        self.device = device
        self.original_path = os.path.abspath(scene_path)
        self.resolved_path = self.parse_scene_path(self.original_path)
        if scene_name is None:
            self.scene_name = self.extract_scene_name(self.resolved_path)
        else:
            assert isinstance(
                scene_name, str
            ), f"scene_name must be str, got {type(scene_name)}"
            self.scene_name = scene_name

        if data_dir is None:
            self.data_dir = self.infer_data_dir(self.resolved_path)
        else:
            assert isinstance(
                data_dir, (str, Path)
            ), f"data_dir must be str or Path, got {type(data_dir)}"
            self.data_dir = str(data_dir)
        self._model: Optional[Any] = None
        self._cache: Optional[CombinedDatasetCache] = cache
        self._cache_key: Optional[str] = cache_key
        self.snapshots: CombinedDatasetCache = self._init_snapshots_cache()

    @staticmethod
    @abstractmethod
    def parse_scene_path(path: str) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def extract_scene_name(resolved_path: str) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def infer_data_dir(resolved_path: str) -> Optional[str]:
        raise NotImplementedError

    @staticmethod
    def _load_cfg_args(scene_path: str) -> Namespace:
        cfg_path = os.path.join(scene_path, 'cfg_args')
        assert os.path.isfile(cfg_path), f"cfg_args not found: {cfg_path}"

        with open(cfg_path, 'r', encoding='utf-8') as handle:
            cfg_text = handle.read()

        cfg = eval(cfg_text)
        assert isinstance(
            cfg, Namespace
        ), f"cfg_args at '{cfg_path}' did not evaluate to Namespace (got {type(cfg)})"

        return cfg

    @staticmethod
    def _infer_data_dir_from_cfg_args(scene_path: str) -> str:
        cfg = BaseSceneModel._load_cfg_args(scene_path=scene_path)

        assert hasattr(
            cfg, 'source_path'
        ), f"cfg_args missing source_path for path '{scene_path}'"
        source_path = cfg.source_path
        assert isinstance(
            source_path, str
        ), f"cfg_args source_path must be string for path '{scene_path}', got {type(source_path)}"
        return os.path.normpath(source_path)

    @property
    def model(self) -> Any:
        self._ensure_model_loaded()
        return self._model

    @abstractmethod
    def _load_model(self) -> Any:
        """Load the underlying scene representation for this modality."""

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        loaded = self._load_model()
        assert loaded is not None, "_load_model must return a model instance"
        self._model = loaded
        if self._cache is not None and self._cache_key is not None:
            cached = self._cache.get(
                cache_filepath=self._cache_key,
                device=self.device,
            )
            if cached is not None:
                assert isinstance(cached, dict), "Cached entry must be a dict"
                assert 'inputs' in cached, "Cached entry missing 'inputs'"
                inputs = cached['inputs']
                assert isinstance(inputs, dict), "Cached inputs must be a dict"
                inputs['model'] = self
                cached['inputs'] = inputs
                self._cache.put(value=cached, cache_filepath=self._cache_key)

    def detach(self) -> "BaseSceneModel":
        self._ensure_model_loaded()
        new_model = apply_tensor_op(method='detach', inputs=self._model)
        return self._clone_with_model(model=new_model, device=self.device)

    def cpu(self) -> "BaseSceneModel":
        self._ensure_model_loaded()
        new_model = apply_tensor_op(method='cpu', inputs=self._model)
        return self._clone_with_model(model=new_model, device=torch.device('cpu'))

    def to(self, device: torch.device) -> "BaseSceneModel":
        assert isinstance(device, torch.device), f"{type(device)=}"
        self._ensure_model_loaded()
        new_model = apply_tensor_op(
            method='to',
            method_kwargs={'device': device},
            inputs=self._model,
        )
        return self._clone_with_model(model=new_model, device=device)

    def _clone_with_model(self, model: Any, device: torch.device) -> "BaseSceneModel":
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__ = dict(self.__dict__)
        clone._model = model
        clone.device = device
        return clone

    def build_static_container(
        self, dataset_name: str, scene_name: str, method_name: str, debugger_enabled: bool
    ) -> html.Div:
        assert isinstance(dataset_name, str), f"{type(dataset_name)=}"
        assert isinstance(scene_name, str), f"{type(scene_name)=}"
        assert isinstance(method_name, str), f"{type(method_name)=}"
        assert isinstance(debugger_enabled, bool), f"{type(debugger_enabled)=}"
        body_placeholder = html.Div(
            [],
            id={
                'type': 'model-body',
                'dataset': dataset_name,
                'scene': scene_name,
                'method': method_name,
            },
        )
        return html.Div(
            [body_placeholder],
            id={
                'type': 'model-container',
                'dataset': dataset_name,
                'scene': scene_name,
                'method': method_name,
            },
            style=three_d_styles.base_container_style(),
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        if 'snapshots' in state:
            state.pop('snapshots')
        if '_cache' in state:
            state.pop('_cache')
        if '_cache_key' in state:
            state.pop('_cache_key')
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.snapshots = self._init_snapshots_cache()
        self._cache = None
        self._cache_key = None

    def _init_snapshots_cache(self) -> CombinedDatasetCache:
        assert isinstance(self.data_dir, str), f"{type(self.data_dir)=}"
        data_root = os.path.abspath(self.data_dir)
        assert os.path.isdir(
            data_root
        ), f"data_dir must be an existing directory, got {data_root}"
        cache_root = os.path.join(data_root, 'render_snapshots')
        version_key = f"{self.resolved_path}::{self.__class__.__name__}"
        version_hash = xxhash.xxh64(version_key.encode()).hexdigest()[:16]
        version_dict = {
            'scene_path': self.resolved_path,
            'scene_model': self.__class__.__name__,
        }
        return CombinedDatasetCache(
            data_root=cache_root,
            version_hash=version_hash,
            use_cpu_cache=True,
            use_disk_cache=True,
            max_cpu_memory_percent=80.0,
            enable_cpu_validation=False,
            enable_disk_validation=False,
            dataset_class_name=self.__class__.__name__,
            version_dict=version_dict,
        )

    def _snapshot_cache_filepath(self, camera_name: str) -> str:
        assert isinstance(camera_name, str), f"{type(camera_name)=}"
        parts = camera_name.split('/')
        assert (
            len(parts) == 2
        ), f"camera_name must be 'images/<filename>', got {camera_name}"
        assert (
            parts[0] == 'images'
        ), f"camera_name must start with 'images/', got {camera_name}"
        filename = parts[1]
        assert filename, f"camera_name missing filename: {camera_name}"
        assert '.' in filename, f"camera_name missing extension: {camera_name}"
        cache_filename = f"{filename}.pt"
        assert isinstance(
            self.snapshots, CombinedDatasetCache
        ), f"{type(self.snapshots)=}"
        return os.path.join(self.snapshots.version_dir, cache_filename)

    def _get_snapshot(self, camera_name: str) -> Optional[torch.Tensor]:
        cache_filepath = self._snapshot_cache_filepath(camera_name)
        cached = self.snapshots.get(
            cache_filepath=cache_filepath,
            device='cpu',
        )
        if cached is None:
            return None
        assert 'inputs' in cached, "Cached snapshot missing inputs"
        inputs = cached['inputs']
        assert isinstance(inputs, dict), f"{type(inputs)=}"
        assert 'snapshot' in inputs, "Cached snapshot missing inputs['snapshot']"
        snapshot = inputs['snapshot']
        assert isinstance(snapshot, torch.Tensor), f"{type(snapshot)=}"
        assert snapshot.device.type == 'cpu', "Snapshot must be on CPU"
        assert not snapshot.requires_grad, "Snapshot must be detached"
        return snapshot

    def _put_snapshot(self, camera_name: str, snapshot: torch.Tensor) -> None:
        assert isinstance(snapshot, torch.Tensor), f"{type(snapshot)=}"
        assert snapshot.device.type == 'cpu', "Snapshot must be on CPU"
        assert not snapshot.requires_grad, "Snapshot must be detached"
        cache_filepath = self._snapshot_cache_filepath(camera_name)
        cache_value = {
            'inputs': {'snapshot': snapshot},
            'labels': {},
            'meta_info': {'camera_name': camera_name},
        }
        self.snapshots.put(
            value=cache_value,
            cache_filepath=cache_filepath,
        )

    @staticmethod
    @abstractmethod
    def register_callbacks(
        dataset: Any, app: dash.Dash, viewer: Any, **kwargs: Any
    ) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def setup_states(app: dash.Dash, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def display_render(
        self,
        camera: Camera,
        resolution: Tuple[int, int],
        camera_name: Optional[str] = None,
        display_cameras: Optional[List[Camera]] = None,
        title: Optional[str] = None,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    @staticmethod
    def _apply_camera_overlays(
        image: torch.Tensor,
        display_cameras: Optional[List[Camera]],
        render_at_camera: Camera,
        resolution: Tuple[int, int],
    ) -> torch.Tensor:
        """Apply camera overlays to rendered image.

        Args:
            image: Rendered image tensor
            display_cameras: Cameras to overlay
            render_at_camera: Render camera
            resolution: Render resolution

        Returns:
            Image with camera overlays applied
        """
        if display_cameras is None:
            return image

        assert isinstance(render_at_camera, Camera), f"{type(render_at_camera)=}"
        target_device = render_at_camera.intrinsics.device
        image = apply_tensor_op(
            method='to',
            method_kwargs={'device': target_device},
            inputs=image,
        )
        composed = image.clone()
        for display_camera in display_cameras:
            assert isinstance(display_camera, Camera), f"{type(display_camera)=}"
            cam_image, cam_mask = render_camera(
                camera=display_camera.to(target_device),
                render_at_camera=render_at_camera,
                render_at_resolution=resolution,
                return_mask=True,
            )
            assert (
                cam_image.device == target_device
            ), f"{cam_image.device=}, {target_device=}"
            assert (
                cam_mask.device == target_device
            ), f"{cam_mask.device=}, {target_device=}"
            indices = cam_mask.nonzero(as_tuple=True)
            composed[:, indices[0], indices[1]] = cam_image[:, indices[0], indices[1]]

        return composed
