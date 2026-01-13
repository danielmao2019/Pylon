"""Step to apply a coordinate transform to cameras and sparse point cloud."""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from data.pipelines.base_step import BaseStep
from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.point_cloud import load_point_cloud, save_point_cloud
from data.structures.three_d.point_cloud.ops.apply_transform import apply_transform
from data.structures.three_d.transforms_json.transforms_json import TransformsJSON


class ApplyCoordTransform(BaseStep):
    """Apply a similarity transform to frames and sparse point cloud."""

    STEP_NAME = "apply_coord_transform"

    def __init__(
        self,
        scene_root: str | Path,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> None:
        super().__init__(input_root=scene_root, output_root=scene_root)
        # Input validation
        assert isinstance(scale, (float, np.floating)), f"{type(scale)=}"
        assert scale > 0, f"{scale=}"
        assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
        assert rotation.shape == (3, 3), f"{rotation.shape=}"
        assert rotation.dtype == np.float32, f"{rotation.dtype=}"
        assert isinstance(translation, np.ndarray), f"{type(translation)=}"
        assert translation.shape == (3,), f"{translation.shape=}"
        assert translation.dtype == np.float32, f"{translation.dtype=}"

        self.scene_root = Path(scene_root)
        self.scale = float(scale)
        self.rotation = rotation
        self.translation = translation

    def _init_input_files(self) -> None:
        self.input_files = ["transforms.json", "sparse_pc.ply"]

    def _init_output_files(self) -> None:
        self.output_files = ["transforms.json", "sparse_pc.ply"]

    def check_outputs(self) -> bool:
        return False

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        self._align_cameras()
        self._align_point_cloud()
        return {}

    def _align_cameras(self) -> None:
        """Load transforms.json, apply similarity transform, and overwrite file."""
        transforms_path = self.scene_root / "transforms.json"
        transforms = TransformsJSON(filepath=transforms_path, device="cpu")
        updated_cameras = []
        for camera in transforms.cameras:
            pose = camera.extrinsics.detach().cpu().numpy()
            R_old = pose[:3, :3]
            t_old = pose[:3, 3]
            R_new = self.rotation @ R_old
            t_new = self.scale * (self.rotation @ t_old) + self.translation
            new_pose = np.eye(4, dtype=np.float32)
            new_pose[:3, :3] = R_new
            new_pose[:3, 3] = t_new
            updated_cameras.append(
                Camera(
                    intrinsics=camera.intrinsics,
                    extrinsics=torch.from_numpy(new_pose),
                    convention=camera.convention,
                    name=camera.name,
                    id=camera.id,
                    device="cpu",
                )
            )
        transforms.cameras = updated_cameras
        TransformsJSON.save(transforms, transforms_path)
        logging.info("   ✓ Updated transforms.json with aligned poses")

    def _align_point_cloud(self) -> str:
        logging.info("🌐 Aligning COLMAP sparse point cloud")
        transforms_path = self.scene_root / "transforms.json"
        transforms = TransformsJSON(filepath=transforms_path, device="cpu")
        ply_file_rel = transforms.ply_file_path
        raw_ply_path = (self.scene_root / ply_file_rel).resolve()
        assert raw_ply_path.is_file(), f"Missing sparse point cloud: {raw_ply_path}"

        point_cloud = load_point_cloud(str(raw_ply_path), device="cuda")
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = self.scale * self.rotation
        transform[:3, 3] = self.translation
        point_cloud.xyz = apply_transform(point_cloud.xyz, transform)
        save_point_cloud(point_cloud, str(raw_ply_path))
        logging.info(
            "   ✓ Applied alignment transform to sparse point cloud: %s", raw_ply_path
        )

        return str(raw_ply_path)
