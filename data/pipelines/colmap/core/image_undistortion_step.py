"""Step that runs COLMAP image undistortion."""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from data.pipelines.base_step import BaseStep
from data.structures.three_d.colmap.load import _load_colmap_images_bin


class ColmapImageUndistortionStep(BaseStep):
    """Copy from NerfStudio pipeline: COLMAP image_undistorter stage."""

    STEP_NAME = "colmap_image_undistortion"

    def __init__(self, scene_root: str | Path) -> None:
        scene_root = Path(scene_root)
        self.input_images_dir = scene_root / "input"
        self.output_images_dir = scene_root / "images"
        self.temp_sparse_dir = scene_root / "sparse"
        self.distorted_sparse_dir = scene_root / "distorted" / "sparse"
        self.undistorted_sparse_dir = scene_root / "undistorted" / "sparse"
        super().__init__(input_root=scene_root, output_root=scene_root)

    def _init_input_files(self) -> None:
        self.image_names = self._get_registered_image_names(
            sparse_root=self.distorted_sparse_dir
        )
        relpaths = [f"input/{name}" for name in self.image_names]
        relpaths.extend(
            [
                "distorted/sparse/0/cameras.bin",
                "distorted/sparse/0/images.bin",
                "distorted/sparse/0/points3D.bin",
            ]
        )
        self.input_files = relpaths

    def _init_output_files(self) -> None:
        relpaths = [f"images/{name}" for name in self.image_names]
        relpaths.extend(
            [
                "undistorted/sparse/0/cameras.bin",
                "undistorted/sparse/0/images.bin",
                "undistorted/sparse/0/points3D.bin",
            ]
        )
        self.output_files = relpaths

    def check_outputs(self) -> bool:
        outputs_ready = super().check_outputs()
        if not outputs_ready:
            return False
        try:
            self._validate_outputs_clean()
            return True
        except Exception as e:
            logging.debug("Image undistortion output validation failed: %s", e)
            return False

    def _validate_outputs_clean(self) -> None:
        target_dir = self.output_root / "images"
        expected_names = self._get_registered_image_names(
            sparse_root=self.undistorted_sparse_dir
        )
        assert expected_names, "No registered images found for undistorted sparse model"
        disk_names = self._get_disk_image_names(images_dir=target_dir)
        assert expected_names == disk_names, (
            "Undistorted images on disk do not match registered images. "
            f"expected={len(expected_names)} actual={len(disk_names)}"
        )

    def build(self, force: bool = False) -> None:
        super().build(force=force)
        self.run(kwargs={}, force=force)

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        if self.check_outputs() and not force:
            return {}

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.undistorted_sparse_dir.mkdir(parents=True, exist_ok=True)

        logging.info("   📐 Image undistortion")
        cmd_parts = self._build_colmap_command()
        result = subprocess.run(cmd_parts, capture_output=True, text=True)
        assert result.returncode == 0, (
            "COLMAP image undistortion failed with code "
            f"{result.returncode}. STDOUT: {result.stdout} STDERR: {result.stderr}"
        )
        self._move_sparse_model()
        self._clean_other_files()

        self._validate_outputs_clean()
        return {}

    def _build_colmap_command(self) -> List[str]:
        return [
            "colmap",
            "image_undistorter",
            "--image_path",
            str(self.input_images_dir),
            "--input_path",
            str(self.distorted_sparse_dir / "0"),
            "--output_path",
            str(self.output_root),
            "--output_type",
            "COLMAP",
        ]

    def _move_sparse_model(self) -> None:
        destination_dir = self.undistorted_sparse_dir / "0"
        source_model_dir = self.temp_sparse_dir
        self._validate_sparse_model_dir(model_dir=source_model_dir)
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        destination_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_model_dir, destination_dir)
        if self.temp_sparse_dir.exists():
            shutil.rmtree(self.temp_sparse_dir)

    def _clean_other_files(self) -> None:
        target_dir = self.output_root / "images"
        expected_names = set(
            self._get_registered_image_names(sparse_root=self.undistorted_sparse_dir)
        )
        assert expected_names, "No registered images found for undistorted sparse model"
        for entry in target_dir.iterdir():
            if entry.is_file() and entry.name not in expected_names:
                entry.unlink()

    def _get_disk_image_names(self, images_dir: Path) -> List[str]:
        assert (
            images_dir.is_dir()
        ), f"Undistorted images directory not found: {images_dir}"
        disk_entries = sorted(images_dir.iterdir())
        assert disk_entries, f"No undistorted images found in {images_dir}"
        assert all(entry.is_file() for entry in disk_entries), (
            f"Non-file entries present in {images_dir}: "
            f"{', '.join(sorted(entry.name for entry in disk_entries if not entry.is_file()))}"
        )
        assert all(entry.suffix == ".png" for entry in disk_entries), (
            f"Non-PNG files present in {images_dir}: "
            f"{', '.join(sorted(entry.name for entry in disk_entries if entry.suffix != '.png'))}"
        )
        return sorted(entry.name for entry in disk_entries)

    def _get_registered_image_names(self, sparse_root: Path) -> List[str]:
        sparse_model_dir = sparse_root / "0"
        if not sparse_model_dir.is_dir():
            return []
        images_bin = sparse_model_dir / "images.bin"
        if not images_bin.is_file():
            return []
        images = _load_colmap_images_bin(path_to_model_file=str(images_bin))
        if not images:
            return []
        image_names = sorted(image.name for image in images.values())
        return image_names

    def _validate_sparse_model_dir(self, model_dir: Path) -> None:
        assert model_dir.is_dir(), f"Sparse model directory not found: {model_dir}"
        expected_files = ["cameras.bin", "images.bin", "points3D.bin"]
        missing_files = [
            name for name in expected_files if not (model_dir / name).exists()
        ]
        assert not missing_files, (
            "Sparse model directory missing expected files: "
            f"{', '.join(missing_files)}"
        )
