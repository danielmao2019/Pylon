import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from data.structures.three_d.camera.camera import Camera
from models.three_d.milef2025learning.render import (
    build_distance_lod_config,
    distance_lod_mask_for_camera,
)
from models.three_d.ordered_gaussians.scene_model import OrderedGaussiansSceneModel
from models.three_d.original_3dgs.loader import load_3dgs_model_original


class Milef2025LearningSceneModel(OrderedGaussiansSceneModel):
    """Scene model for Milef2025Learning checkpoints with distance-based LOD."""

    DISTANCE_DMIN: float = 1.0
    DISTANCE_DMAX: float = 5.0
    N_HIGH_FRACTION: float = 1.0
    N_LOW_FRACTION: float = 0.1

    def __init__(
        self,
        scene_path: str,
        device: Optional[torch.device] = None,
        cache: Optional[Any] = None,
        cache_key: Optional[str] = None,
    ) -> None:
        self._distance_lod_cfg: Optional[Dict[str, float | int]] = None
        super().__init__(
            scene_path=scene_path,
            device=device,
            cache=cache,
            cache_key=cache_key,
        )

    def _load_model(self) -> Any:
        model = load_3dgs_model_original(
            model_dir=self.resolved_path,
            device=self.device,
            iteration=30_000,
        )
        total = int(model.get_xyz.shape[0])
        assert total > 0, "Milef2025Learning model must contain Gaussians"
        self._gaussians_total = total
        self._distance_lod_cfg = build_distance_lod_config(
            total_gaussians=total,
            dmin=self.DISTANCE_DMIN,
            dmax=self.DISTANCE_DMAX,
            n_high_fraction=self.N_HIGH_FRACTION,
            n_low_fraction=self.N_LOW_FRACTION,
        )
        return model

    @staticmethod
    def parse_scene_path(path: str) -> str:
        # Input validations
        assert isinstance(path, str), f"{type(path)=}"
        assert os.path.isdir(
            path
        ), f"Expected existing directory for Milef2025Learning scene, got '{path}'"

        resolved_path = os.path.abspath(path)
        cfg_path = os.path.join(resolved_path, 'cfg_args')
        assert os.path.isfile(cfg_path), f"cfg_args not found at '{cfg_path}'"
        cfg = OrderedGaussiansSceneModel._load_cfg_args(scene_path=resolved_path)
        required_fields = {
            'data_device',
            'depths',
            'eval',
            'images',
            'model_path',
            'resolution',
            'sh_degree',
            'source_path',
            'train_test_exp',
            'white_background',
            'method',
            'point_cloud_dir',
        }
        actual_fields = set(vars(cfg).keys())
        missing_fields = required_fields - actual_fields
        assert not missing_fields, f"cfg_args missing required fields {missing_fields}"
        unexpected_fields = actual_fields - required_fields
        assert (
            not unexpected_fields
        ), f"cfg_args contains unexpected fields {unexpected_fields}"
        assert (
            cfg.method == "milef2025learning"
        ), f"cfg_args method must be 'milef2025learning', got {cfg.method!r}"

        ply_path = os.path.join(
            resolved_path,
            'point_cloud',
            'iteration_30000',
            'point_cloud.ply',
        )
        assert os.path.isfile(
            ply_path
        ), f"point_cloud.ply missing for iteration 30000: {ply_path}"

        cameras_path = os.path.join(resolved_path, 'cameras.json')
        assert os.path.isfile(cameras_path), f"cameras.json not found: {cameras_path}"

        return resolved_path

    @staticmethod
    def extract_scene_name(resolved_path: str) -> str:
        base = Path(resolved_path).resolve().name
        parts = base.split('-')
        if len(parts) >= 6:
            return '-'.join(parts[:6])
        return base

    @staticmethod
    def infer_data_dir(resolved_path: str) -> Optional[str]:
        return OrderedGaussiansSceneModel._infer_data_dir_from_cfg_args(resolved_path)

    def _compute_cutoff_count(
        self,
        camera: Camera,
        resolution: Tuple[int, int],
        device: torch.device,
    ) -> Tuple[int, Optional[torch.Tensor]]:
        # Input validations
        assert self._distance_lod_cfg is not None, "Distance LOD config must be set"

        mask = distance_lod_mask_for_camera(
            positions=self.model.get_xyz,
            camera=camera,
            resolution=resolution,
            device=device,
            config=self._distance_lod_cfg,
        )
        count = int(mask.sum().item())
        assert count > 0, "Distance LOD mask produced zero Gaussians"
        return count, mask
