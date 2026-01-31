"""Step to extract COLMAP camera intrinsics and extrinsics."""

import logging
from pathlib import Path
from typing import Any, Dict

import torch

from data.pipelines.base_step import BaseStep
from data.structures.three_d.colmap.colmap_data import COLMAP_Data
from data.structures.three_d.colmap.convert import create_nerfstudio_from_colmap
from data.structures.three_d.nerfstudio.nerfstudio_data import NerfStudio_Data


class ColmapExtractCamerasStep(BaseStep):
    """Export complete NerfStudio_Data JSON from COLMAP outputs."""

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
            nerfstudio = NerfStudio_Data.load(
                filepath=self.transforms_path, device=torch.device("cpu")
            )
            frame_names = [Path(name).name for name in nerfstudio.filenames]
            disk_names = self._validate_disk_images()
            assert set(frame_names) == disk_names, (
                "Frame file_paths do not match undistorted images on disk. "
                f"frames={len(frame_names)} disk={len(disk_names)}"
            )
        except Exception as e:
            logging.debug("COLMAP NerfStudio_Data validation failed: %s", e)
            return False
        return True

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        self.output_root.mkdir(parents=True, exist_ok=True)
        if not force and self.check_outputs():
            logging.info("🎥 COLMAP cameras already extracted - SKIPPED")
            return {}

        colmap_data = COLMAP_Data(model_dir=self.model_dir)
        create_nerfstudio_from_colmap(
            filename="transforms.json",
            colmap_cameras=colmap_data.cameras,
            colmap_images=colmap_data.images,
            output_dir=str(self.output_root),
            ply_file_path="sparse_pc.ply",
        )
        logging.info(
            "   ✓ Wrote transforms.json with %d frames", len(colmap_data.images)
        )
        return {}

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
