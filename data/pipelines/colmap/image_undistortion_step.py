"""Step that runs COLMAP image undistortion."""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict

from data.pipelines.base_step import BaseStep
from utils.io.colmap.load_colmap import load_model


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
        entries = sorted(self.input_images_dir.iterdir())
        assert entries, f"Empty input dir or no files: {self.input_images_dir}"
        assert all(entry.is_file() for entry in entries), (
            "COLMAP input directory must only contain files "
            f"(found non-file entries in {self.input_images_dir})"
        )
        self.image_names = [entry.name for entry in entries]
        filenames = [f"input/{name}" for name in self.image_names]
        filenames.extend(
            [
                "distorted/sparse/0/cameras.bin",
                "distorted/sparse/0/images.bin",
                "distorted/sparse/0/points3D.bin",
            ]
        )
        self.input_files = filenames

    def _init_output_files(self) -> None:
        self.output_files = [f"images/{name}" for name in self.image_names]
        self.output_files.extend(
            [
                "undistorted/sparse/0/cameras.bin",
                "undistorted/sparse/0/images.bin",
                "undistorted/sparse/0/points3D.bin",
            ]
        )

    def check_outputs(self) -> bool:
        outputs_ready = super().check_outputs()
        if not outputs_ready:
            return False
        try:
            self._assert_outputs_clean()
            self._assert_sparse_model_matches_images()
            return True
        except Exception as e:
            logging.debug("Image undistortion output validation failed: %s", e)
            return False

    def _assert_outputs_clean(self) -> None:
        target_dir = self.output_root / "images"
        assert (
            target_dir.is_dir()
        ), f"Undistorted images directory not found: {target_dir}"
        children = list(target_dir.iterdir())
        assert all(entry.is_file() for entry in children), (
            f"Non-file entries present in {target_dir}: "
            f"{', '.join(sorted(entry.name for entry in children if not entry.is_file()))}"
        )
        assert all(entry.suffix == ".png" for entry in children), (
            f"Non-PNG files present in {target_dir}: "
            f"{', '.join(sorted(entry.name for entry in children if entry.suffix != '.png'))}"
        )
        disk_names = sorted(entry.name for entry in children)
        assert disk_names == sorted(self.image_names), (
            "Undistorted images on disk do not match inputs. "
            f"expected={len(self.image_names)} actual={len(disk_names)}"
        )

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.undistorted_sparse_dir.mkdir(parents=True, exist_ok=True)
        if not force and self.check_outputs():
            logging.info("📐 COLMAP undistortion already done - SKIPPED")
            return {}
        logging.info("   📐 Image undistortion")
        undistort_cmd = (
            f"colmap image_undistorter "
            f"--image_path {self.input_images_dir} "
            f"--input_path {self.distorted_sparse_dir / '0'} "
            f"--output_path {self.output_root} "
            f"--output_type COLMAP"
        )
        ret_code = subprocess.call(undistort_cmd, shell=True)
        assert ret_code == 0, f"COLMAP image undistortion failed with code {ret_code}"
        self._move_sparse_model()
        self._assert_sparse_model_matches_images()
        self._clean_other_files()
        return {}

    def _assert_sparse_model_matches_images(self) -> None:
        sparse_model_dir = self.undistorted_sparse_dir / "0"
        assert (
            sparse_model_dir.is_dir()
        ), f"Sparse model directory not found: {sparse_model_dir}"
        _, images, _ = load_model(str(sparse_model_dir))
        assert images, f"No registered images parsed from {sparse_model_dir}"
        image_names = sorted(image.name for image in images.values())
        assert all(name.endswith(".png") for name in image_names), (
            "Sparse model contains non-PNG image names: "
            f"{', '.join(sorted(name for name in image_names if not name.endswith('.png')))}"
        )
        disk_entries = sorted(self.output_images_dir.iterdir())
        assert disk_entries, f"No undistorted images found in {self.output_images_dir}"
        disk_names = sorted(
            entry.name
            for entry in disk_entries
            if entry.is_file() and entry.suffix == ".png"
        )
        assert image_names == disk_names, (
            "Sparse model image names do not match undistorted images on disk. "
            f"sparse_count={len(image_names)} disk_count={len(disk_names)}"
        )

    def _move_sparse_model(self) -> None:
        destination_dir = self.undistorted_sparse_dir / "0"
        source_model_dir = self.temp_sparse_dir / "0"
        if not source_model_dir.is_dir():
            assert (
                self.temp_sparse_dir.is_dir()
            ), f"Sparse model directory not found: {self.temp_sparse_dir}"
            source_model_dir = self.temp_sparse_dir
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        destination_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_model_dir, destination_dir)
        if self.temp_sparse_dir.exists():
            shutil.rmtree(self.temp_sparse_dir)

    def _clean_other_files(self) -> None:
        target_dir = self.output_root / "images"
        assert (
            target_dir.is_dir()
        ), f"Undistorted images directory not found: {target_dir}"
        children = list(target_dir.iterdir())
        assert all(entry.is_file() for entry in children), (
            f"Non-file entries present in {target_dir}: "
            f"{', '.join(sorted(entry.name for entry in children if not entry.is_file()))}"
        )
        source_entries = sorted(self.input_images_dir.iterdir())
        expected_names = {entry.name for entry in source_entries}
        for entry in children:
            if entry.name in expected_names:
                continue
            entry.unlink()
        remaining = {entry.name for entry in target_dir.iterdir()}
        assert remaining == expected_names, (
            f"Unexpected files remain in {target_dir}: "
            f"{', '.join(sorted(remaining - expected_names))}"
        )
