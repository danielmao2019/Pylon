"""Base step for initializing a COLMAP model from external pose priors."""

import sqlite3
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from data.pipelines.base_step import BaseStep
from data.structures.three_d.colmap.colmap_data import COLMAP_Data
from data.structures.three_d.colmap.load import (
    CAMERA_MODELS,
    ColmapCamera,
    ColmapImage,
    ColmapPoint3D,
)


class BaseColmapInitStep(BaseStep):
    """Generic COLMAP initialization from externally provided poses."""

    STEP_NAME = "base_colmap_init"

    def __init__(self, scene_root: str | Path) -> None:
        scene_root = Path(scene_root)
        self.scene_root = scene_root
        self.input_images_dir = scene_root / "input"
        self.database_path = scene_root / "distorted" / "database.db"
        self.model_dir = scene_root / "colmap_init"
        super().__init__(input_root=scene_root, output_root=scene_root)

    def _init_input_files(self) -> None:
        image_entries = sorted(self.input_images_dir.glob("*.png"))
        self.image_names = [entry.name for entry in image_entries]
        filenames = ["input"]
        filenames.extend([f"input/{name}" for name in self.image_names])
        filenames.append("distorted/database.db")
        self.input_files = filenames

    def _init_output_files(self) -> None:
        self.output_files = [
            "colmap_init/cameras.bin",
            "colmap_init/images.bin",
            "colmap_init/points3D.bin",
            "colmap_init/cameras.txt",
            "colmap_init/images.txt",
            "colmap_init/points3D.txt",
        ]

    def build(self, force: bool = False) -> None:
        super().build(force=force)
        self.run(kwargs={}, force=force)

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        if not force and self.check_outputs():
            return {}
        self.model_dir.mkdir(parents=True, exist_ok=True)
        camera_id, camera_model, size, params = self._load_cameras_from_database()
        poses = self._define_poses()
        colmap_cameras, colmap_images, colmap_points = self._build_colmap_model(
            camera_id=camera_id,
            camera_model=camera_model,
            size=size,
            params=params,
            poses=poses,
        )
        colmap_data = COLMAP_Data(
            cameras=colmap_cameras,
            images=colmap_images,
            points3D=colmap_points,
        )
        colmap_data.save(output_dir=self.model_dir)
        return {}

    @abstractmethod
    def _define_poses(self) -> List[Dict[str, Any]]:
        """Return pose dicts with keys: name, qvec, tvec, image_id."""
        raise NotImplementedError("Subclasses must implement _define_poses()")

    def _load_image_ids_from_database(self) -> Dict[str, int]:
        # Input validations
        assert (
            self.database_path.is_file()
        ), f"COLMAP database missing at {self.database_path}"

        connection = sqlite3.connect(self.database_path)
        cursor = connection.cursor()
        rows = cursor.execute("SELECT image_id, name FROM images").fetchall()
        connection.close()
        assert rows, f"No images found in database {self.database_path}"

        image_id_by_name = {row[1]: int(row[0]) for row in rows}
        assert len(image_id_by_name) == len(rows), (
            "Duplicate image names found in database " f"{self.database_path}"
        )
        return image_id_by_name

    def _load_cameras_from_database(
        self,
    ) -> Tuple[int, str, Tuple[int, int], np.ndarray]:
        connection = sqlite3.connect(self.database_path)
        cursor = connection.cursor()
        rows = cursor.execute(
            "SELECT camera_id, model, width, height, params FROM cameras"
        ).fetchall()
        connection.close()
        assert rows, f"No camera rows found in database {self.database_path}"
        assert (
            len(rows) == 1
        ), f"Expected exactly one camera row in {self.database_path}, got {len(rows)}"
        row = rows[0]
        camera_id = int(row[0])
        model_id = int(row[1])
        assert (
            model_id in CAMERA_MODELS
        ), f"Unknown COLMAP camera model id {model_id} in {self.database_path}"
        model = CAMERA_MODELS[model_id].model_name
        expected_param_count = CAMERA_MODELS[model_id].num_params
        width = int(row[2])
        height = int(row[3])
        assert (
            width > 0 and height > 0
        ), f"Camera dimensions must be positive, got width={width}, height={height}"
        params = np.frombuffer(row[4], dtype=np.float64)
        assert params.size == expected_param_count, (
            f"Camera params blob length mismatch: expected {expected_param_count} "
            f"got {params.size}"
        )
        return camera_id, model, (width, height), params

    def _build_colmap_model(
        self,
        camera_id: int,
        camera_model: str,
        size: Tuple[int, int],
        params: np.ndarray,
        poses: List[Dict[str, Any]],
    ) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
        # Input validations
        assert isinstance(poses, list), f"{type(poses)=}"
        assert poses, "poses must be non-empty"
        assert all(isinstance(pose, dict) for pose in poses), f"{poses=}"
        assert all("name" in pose for pose in poses), "pose missing name"
        assert all("qvec" in pose for pose in poses), "pose missing qvec"
        assert all("tvec" in pose for pose in poses), "pose missing tvec"
        assert all("image_id" in pose for pose in poses), "pose missing image_id"
        assert all(
            isinstance(pose["image_id"], (int, np.integer)) for pose in poses
        ), f"{[pose['image_id'] for pose in poses]=}"

        width, height = size
        colmap_cameras = {
            camera_id: ColmapCamera(
                id=camera_id,
                model=camera_model,
                width=width,
                height=height,
                params=params,
            )
        }
        image_ids = [int(pose["image_id"]) for pose in poses]
        assert len(set(image_ids)) == len(
            image_ids
        ), "Duplicate image_id values in pose list"
        colmap_images: Dict[int, ColmapImage] = {
            int(pose["image_id"]): ColmapImage(
                id=int(pose["image_id"]),
                qvec=pose["qvec"],
                tvec=pose["tvec"],
                camera_id=camera_id,
                name=pose["name"],
                xys=np.empty((0, 2), dtype=np.float64),
                point3D_ids=np.empty((0,), dtype=np.int64),
            )
            for pose in poses
        }
        colmap_points: Dict[int, ColmapPoint3D] = {}
        return colmap_cameras, colmap_images, colmap_points
