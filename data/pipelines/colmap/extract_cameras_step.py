"""Step to extract COLMAP camera intrinsics and extrinsics."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from data.pipelines.base_step import BaseStep
from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.transforms_json.transforms_json import TransformsJSON
from utils.io.colmap.load_colmap import Camera as ColmapCamera
from utils.io.colmap.load_colmap import (
    Image,
    load_model,
)
from utils.io.json import load_json, save_json
from utils.three_d.rotation.quaternion import qvec2rotmat


class ColmapExtractCamerasStep(BaseStep):
    """Export complete NeRF Studio transforms.json from COLMAP outputs."""

    STEP_NAME = "colmap_extract_cameras"

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        super().__init__(input_root=input_root, output_root=output_root)
        self.model_dir = self.input_root / "0"
        self.transforms_path = self.output_root / "transforms.json"

    def _init_input_files(self) -> None:
        self.input_files = ["0/cameras.bin", "0/images.bin"]

    def _init_output_files(self) -> None:
        self.output_files = ["transforms.json"]

    def check_outputs(self) -> bool:
        outputs_ready = super().check_outputs()
        if not outputs_ready:
            return False
        try:
            transforms = TransformsJSON(
                filepath=self.transforms_path, device=torch.device("cpu")
            )
            frame_names = [Path(name).name for name in transforms.filenames]
            disk_names = self._validate_disk_images()
            assert set(frame_names) == disk_names, (
                "Frame file_paths do not match undistorted images on disk. "
                f"frames={len(frame_names)} disk={len(disk_names)}"
            )
        except Exception as e:
            logging.debug("COLMAP transforms validation failed: %s", e)
            return False
        return True

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        self.output_root.mkdir(parents=True, exist_ok=True)
        if not force and self.check_outputs():
            logging.info("🎥 COLMAP cameras already extracted - SKIPPED")
            return {}

        colmap_cameras, colmap_images, _ = load_model(str(self.model_dir))
        intrinsic_params = self._extract_intrinsics(colmap_cameras)
        cameras = self._extract_cameras(colmap_images, intrinsic_params)
        self._build_transforms_json(intrinsic_params=intrinsic_params, cameras=cameras)
        logging.info("   ✓ Wrote transforms.json with %d frames", len(cameras))
        return {}

    def _extract_intrinsics(self, cameras: Dict[int, ColmapCamera]) -> Dict[str, Any]:
        assert cameras, "No cameras found in COLMAP model"
        assert len(cameras) == 1, f"Expected exactly one camera, got {len(cameras)}"
        camera = next(iter(cameras.values()))
        params = camera.params
        model = camera.model
        assert model == "PINHOLE", f"Expected COLMAP camera model PINHOLE, got {model}"
        width = camera.width
        height = camera.height
        assert isinstance(
            width, (int, np.integer)
        ), f"Expected integer camera width, got {type(width)}"
        assert isinstance(
            height, (int, np.integer)
        ), f"Expected integer camera height, got {type(height)}"
        assert (
            width > 0 and height > 0
        ), f"Camera dimensions must be positive, got width={width} height={height}"
        intrinsic_params: Dict[str, Any] = {
            "w": int(width),
            "h": int(height),
            "fl_x": float(params[0]),
            "fl_y": float(params[1]),
            "cx": float(params[2]),
            "cy": float(params[3]),
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "camera_model": "OPENCV",
        }
        return intrinsic_params

    def _extract_cameras(
        self, images: Dict[int, Image], intrinsic_params: Dict[str, Any]
    ) -> List[Camera]:
        assert images, "No images available in COLMAP model"
        cameras: List[Camera] = []
        intrinsics = torch.tensor(
            [
                [intrinsic_params["fl_x"], 0.0, intrinsic_params["cx"]],
                [0.0, intrinsic_params["fl_y"], intrinsic_params["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        for image_id, image in sorted(images.items()):
            rotation = qvec2rotmat(image.qvec)
            translation = image.tvec.reshape(3, 1)
            world_to_camera = np.concatenate([rotation, translation], axis=1)
            world_to_camera = np.concatenate(
                [world_to_camera, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0
            )
            camera_to_world = np.linalg.inv(world_to_camera)
            extrinsics_opencv = torch.from_numpy(camera_to_world).to(torch.float32)
            camera = Camera(
                intrinsics=intrinsics,
                extrinsics=extrinsics_opencv,
                convention="opencv",
                name=Path(image.name).stem,
                id=image_id,
                device=extrinsics_opencv.device,
            ).to(convention="opengl")
            cameras.append(camera)
        return cameras

    @property
    def _default_applied_transform(self) -> List[List[float]]:
        m = np.eye(4)[:3, :]
        m = m[np.array([0, 2, 1]), :]
        m[2, :] *= -1
        return m.tolist()

    def _build_transforms_json(
        self,
        intrinsic_params: Dict[str, Any],
        cameras: List[Camera],
    ) -> None:
        assert cameras, "No cameras provided to build transforms.json"
        payload: Dict[str, Any] = dict(intrinsic_params)
        payload["ply_file_path"] = "sparse_pc.ply"
        payload["applied_transform"] = self._default_applied_transform
        payload["cameras"] = cameras
        TransformsJSON.save(payload, self.transforms_path)

    def _validate_disk_images(self) -> set[str]:
        images_dir = self.output_root / "images"
        assert (
            images_dir.is_dir()
        ), f"Undistorted images directory not found: {images_dir}"
        disk_images = sorted(
            entry.name for entry in images_dir.iterdir() if entry.is_file()
        )
        assert disk_images, f"No images found in {images_dir}"
        disk_names = set(disk_images)
        assert len(disk_names) == len(disk_images), (
            "Duplicate image filenames present on disk: "
            f"{', '.join(sorted(name for name in disk_images if disk_images.count(name) > 1))}"
        )
        assert all(
            name.endswith(".png") for name in disk_names
        ), f"Non-PNG images present in {images_dir}"
        return disk_names
