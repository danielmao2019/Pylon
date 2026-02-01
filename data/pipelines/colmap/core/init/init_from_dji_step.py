"""Step that seeds COLMAP with DJI GPS/gimbal initialization."""

import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyproj
import torch

from data.pipelines.base_step import BaseStep
from data.structures.three_d.camera.rotation.euler import euler_to_matrix
from data.structures.three_d.camera.rotation.quaternion import rotmat2qvec
from data.structures.three_d.colmap.colmap_data import COLMAP_Data
from data.structures.three_d.colmap.load import (
    CAMERA_MODELS,
    ColmapCamera,
    ColmapImage,
    ColmapPoint3D,
)
from project.data.structures.dji_data import DJI_Data


class ColmapInitFromDJIStep(BaseStep):
    """Build a COLMAP model using GPS+gimbal priors from DJI JPG metadata."""

    STEP_NAME = "colmap_init_from_dji"

    def __init__(self, scene_root: str | Path, dji_data_root: str | Path) -> None:
        scene_root = Path(scene_root)
        self.scene_root = scene_root
        self.dji_data_root = Path(dji_data_root).expanduser().resolve()
        self.input_images_dir = scene_root / "input"
        self.database_path = scene_root / "distorted" / "database.db"
        self.model_dir = scene_root / "distorted" / "init_model"
        self.proj = pyproj.Transformer.from_crs(4326, 32617, always_xy=True)
        super().__init__(input_root=scene_root, output_root=scene_root)

    def _init_input_files(self) -> None:
        entries = sorted(self.input_images_dir.iterdir())
        assert entries, f"Empty input dir or no files: {self.input_images_dir}"
        assert all(entry.is_file() for entry in entries), (
            "COLMAP input directory must only contain files "
            f"(found non-file entries in {self.input_images_dir})"
        )
        self.image_names = [entry.name for entry in entries]
        filenames = [f"input/{name}" for name in self.image_names]
        filenames.append("distorted/database.db")
        self.input_files = filenames

    def _init_output_files(self) -> None:
        self.output_files = [
            "distorted/init_model/cameras.bin",
            "distorted/init_model/images.bin",
            "distorted/init_model/points3D.bin",
            "distorted/init_model/cameras.txt",
            "distorted/init_model/images.txt",
            "distorted/init_model/points3D.txt",
        ]

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        assert (
            self.dji_data_root.is_dir()
        ), f"DJI data root not found: {self.dji_data_root}"
        if not force and self.check_outputs():
            return {}
        self.model_dir.mkdir(parents=True, exist_ok=True)
        camera_id, camera_model, size, params = self._load_camera_from_database()
        dji_jpg_paths = collect_lidar_jpg_paths(
            scene_root=self.dji_data_root, scene_name=self.dji_data_root.name
        )
        png_stems = [Path(name).stem for name in self.image_names]
        assert len(png_stems) == len(dji_jpg_paths), (
            "PNG/JPG counts mismatch when building DJI init model: "
            f"png={len(png_stems)} jpg={len(dji_jpg_paths)}"
        )
        png_to_jpg = self._map_png_to_jpg(
            png_stems=png_stems, dji_jpg_paths=dji_jpg_paths
        )
        poses = self._compute_poses(png_to_jpg=png_to_jpg)
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

    def _load_camera_from_database(
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
        params = np.frombuffer(row[4], dtype=np.float32)
        assert params.size == expected_param_count, (
            f"Camera params blob length mismatch: expected {expected_param_count} "
            f"got {params.size}"
        )
        return camera_id, model, (width, height), params

    def _map_png_to_jpg(
        self, png_stems: List[str], dji_jpg_paths: List[Path]
    ) -> Dict[str, Path]:
        mapping: Dict[str, Path] = {}
        for jpg_path in dji_jpg_paths:
            stem = jpg_path.stem
            assert (
                stem not in mapping
            ), f"Duplicate JPG stem detected during mapping: {stem}"
            mapping[stem] = jpg_path
        for stem in png_stems:
            assert stem in mapping, f"Missing JPG for PNG stem {stem}"
        assert len(mapping) == len(png_stems), (
            "Mismatch between PNG stems and mapped JPGs: "
            f"png_stems={len(png_stems)} mapped_jpgs={len(mapping)}"
        )
        return mapping

    def _compute_poses(self, png_to_jpg: Dict[str, Path]) -> List[Dict[str, Any]]:
        poses: List[Dict[str, Any]] = []
        for image_name in self.image_names:
            stem = Path(image_name).stem
            jpg_path = png_to_jpg[stem]
            tags = extract_dji_gps_gimbal_floats(jpg_path=jpg_path)
            lon = tags["lon"]
            lat = tags["lat"]
            alt = tags["alt"]
            yaw = tags["yaw"]
            pitch = tags["pitch"]
            roll = tags["roll"]
            position = np.array(self.proj.transform(lon, lat, alt), dtype=np.float32)
            angles = torch.tensor(
                [
                    math.radians(roll),
                    math.radians(pitch),
                    math.radians(yaw),
                ],
                dtype=torch.float32,
            )
            camera_to_world = (
                euler_to_matrix(angles).detach().cpu().numpy().astype(np.float32)
            )
            world_to_camera = camera_to_world.T
            qvec = rotmat2qvec(world_to_camera)
            tvec = -world_to_camera @ position
            poses.append(
                {
                    "name": image_name,
                    "qvec": qvec,
                    "tvec": tvec,
                }
            )
        return poses

    def _build_colmap_model(
        self,
        camera_id: int,
        camera_model: str,
        size: Tuple[int, int],
        params: np.ndarray,
        poses: List[Dict[str, Any]],
    ) -> Tuple[
        Dict[int, ColmapCamera],
        Dict[int, ColmapImage],
        Dict[int, ColmapPoint3D],
    ]:
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
        colmap_images: Dict[int, ColmapImage] = {}
        for idx, pose in enumerate(poses, start=1):
            colmap_images[idx] = ColmapImage(
                id=idx,
                qvec=pose["qvec"],
                tvec=pose["tvec"],
                camera_id=camera_id,
                name=pose["name"],
                xys=np.empty((0, 2), dtype=np.float64),
                point3D_ids=np.empty((0,), dtype=np.int64),
            )
        colmap_points: Dict[int, ColmapPoint3D] = {}
        return colmap_cameras, colmap_images, colmap_points
