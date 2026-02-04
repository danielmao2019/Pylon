"""Step that runs COLMAP point_triangulator using DJI-initialized poses."""

import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from data.pipelines.base_step import BaseStep
from data.structures.three_d.colmap.load import (
    _load_colmap_cameras_bin,
    _load_colmap_images_bin,
    _load_colmap_points_bin,
)

MIN_SEED_POINTS = 200


class SeedTooWeakError(RuntimeError):
    """Raised when the seed model is too weak to expand."""


class ColmapPointTriangulationStep(BaseStep):
    """Triangulate points starting from an initialized sparse model."""

    STEP_NAME = "colmap_point_triangulation"

    def __init__(self, scene_root: str | Path) -> None:
        # Input validations
        assert isinstance(scene_root, (str, Path)), f"{type(scene_root)=}"

        # Input normalizations
        scene_root = Path(scene_root)

        self.scene_root = scene_root
        self.input_images_dir = scene_root / "input"
        self.database_path = scene_root / "distorted" / "database.db"
        self.init_model_dir = scene_root / "colmap_init"
        self.triangulated_model_dir: Path | None = None
        self.seed_image_ids: List[int] | None = None
        super().__init__(input_root=scene_root, output_root=scene_root)

    def _init_input_files(self) -> None:
        entries = sorted(self.input_images_dir.iterdir())
        assert entries, f"Empty input dir or no files: {self.input_images_dir}"
        assert all(entry.is_file() for entry in entries), (
            "COLMAP input directory must only contain files "
            f"(found non-file entries in {self.input_images_dir})"
        )
        self.image_names = [entry.name for entry in entries]
        filenames: List[str] = [f"input/{name}" for name in self.image_names]
        filenames.append("distorted/database.db")
        filenames.extend(
            [
                "colmap_init/cameras.bin",
                "colmap_init/images.bin",
                "colmap_init/points3D.bin",
            ]
        )
        self.input_files = filenames

    def _init_output_files(self) -> None:
        seed_image_ids = self._load_seed_image_ids()
        seed_count = len(seed_image_ids)
        self.seed_image_ids = seed_image_ids
        self.triangulated_model_dir = self.scene_root / f"seed{seed_count}_triangulated"
        self.output_files = [
            f"seed{seed_count}_triangulated/cameras.bin",
            f"seed{seed_count}_triangulated/images.bin",
            f"seed{seed_count}_triangulated/points3D.bin",
        ]

    def build(self, force: bool = False) -> None:
        super().build(force=force)
        self.run(kwargs={}, force=force)

    def check_outputs(self) -> bool:
        outputs_ready = super().check_outputs()
        if not outputs_ready:
            return False
        try:
            self._validate_outputs()
            seed_points = self._seed_point_count()
            if seed_points < MIN_SEED_POINTS:
                raise SeedTooWeakError(
                    f"seed_points={seed_points} < {MIN_SEED_POINTS}; "
                    f"seeds={self._load_seed_image_ids()}"
                )
            return True
        except Exception as e:
            return False

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        if not force and self.check_outputs():
            return {}

        assert self.triangulated_model_dir is not None, "triangulated_model_dir unset"
        self.triangulated_model_dir.mkdir(parents=True, exist_ok=True)

        cmd_parts = self._build_colmap_command()
        result = subprocess.run(cmd_parts, capture_output=True, text=True)
        ret_code = result.returncode
        assert ret_code == 0, (
            f"COLMAP point_triangulator failed with code {ret_code}. "
            f"STDOUT: {result.stdout} STDERR: {result.stderr}"
        )

        self._validate_outputs()
        metrics = self._run_model_analyzer()
        logging.info(
            "Seed triangulation model_analyzer: images=%s points=%s observations=%s "
            "mean_reprojection_error_px=%s",
            metrics["images"],
            metrics["points"],
            metrics["observations"],
            metrics["mean_reprojection_error_px"],
        )
        if metrics["points"] < MIN_SEED_POINTS:
            raise SeedTooWeakError(
                f"seed_points={metrics['points']} < {MIN_SEED_POINTS}; "
                f"seeds={self._load_seed_image_ids()}"
            )
        return {}

    def _build_colmap_command(self) -> List[str]:
        assert self.triangulated_model_dir is not None, "triangulated_model_dir unset"
        return [
            "colmap",
            "point_triangulator",
            "--database_path",
            str(self.database_path),
            "--image_path",
            str(self.input_images_dir),
            "--input_path",
            str(self.init_model_dir),
            "--output_path",
            str(self.triangulated_model_dir),
            "--clear_points",
            "1",
            "--refine_intrinsics",
            "0",
            "--log_to_stderr",
            "1",
        ]

    def _validate_outputs(self) -> None:
        self._validate_colmap_cameras()
        self._validate_colmap_images()
        self._validate_colmap_points()

    def _validate_colmap_cameras(self) -> Dict[int, Any]:
        assert self.triangulated_model_dir is not None, "triangulated_model_dir unset"
        cameras_path = self.triangulated_model_dir / "cameras.bin"
        cameras = _load_colmap_cameras_bin(path_to_model_file=str(cameras_path))
        assert cameras, f"No cameras parsed from {cameras_path}"
        assert (
            len(cameras) == 1
        ), f"Expected exactly one camera in {cameras_path}, found {len(cameras)}"
        return cameras

    def _validate_colmap_images(self) -> Dict[int, Any]:
        assert self.triangulated_model_dir is not None, "triangulated_model_dir unset"
        images_path = self.triangulated_model_dir / "images.bin"
        images = _load_colmap_images_bin(path_to_model_file=str(images_path))
        assert images, f"No registered images parsed from {images_path}"
        registered_names = sorted(image.name for image in images.values())
        init_images_path = self.init_model_dir / "images.bin"
        init_images = _load_colmap_images_bin(path_to_model_file=str(init_images_path))
        assert init_images, f"No images parsed from init model {init_images_path}"
        init_names = sorted(image.name for image in init_images.values())
        assert registered_names == init_names, (
            "Triangulated model image names do not match init model images. "
            f"expected={len(init_names)} actual={len(registered_names)}"
        )
        return images

    def _validate_colmap_points(self) -> Dict[int, Any]:
        assert self.triangulated_model_dir is not None, "triangulated_model_dir unset"
        points_path = self.triangulated_model_dir / "points3D.bin"
        points3d = _load_colmap_points_bin(path_to_model_file=str(points_path))
        return points3d

    def _load_seed_image_ids(self) -> List[int]:
        init_images_path = self.init_model_dir / "images.bin"
        images = _load_colmap_images_bin(path_to_model_file=str(init_images_path))
        seed_image_ids = sorted(int(image_id) for image_id in images.keys())
        assert seed_image_ids, f"No images parsed from {init_images_path}"
        return seed_image_ids

    def _seed_point_count(self) -> int:
        assert self.triangulated_model_dir is not None, "triangulated_model_dir unset"
        points_path = self.triangulated_model_dir / "points3D.bin"
        points3d = _load_colmap_points_bin(path_to_model_file=str(points_path))
        return len(points3d)

    def _run_model_analyzer(self) -> Dict[str, Any]:
        assert self.triangulated_model_dir is not None, "triangulated_model_dir unset"
        cmd_parts = [
            "colmap",
            "model_analyzer",
            "--path",
            str(self.triangulated_model_dir),
        ]
        result = subprocess.run(cmd_parts, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"COLMAP model_analyzer failed with code {result.returncode}. "
            f"STDOUT: {result.stdout} STDERR: {result.stderr}"
        )
        output = f"{result.stdout}\n{result.stderr}"
        return self._parse_model_analyzer_output(output=output)

    def _parse_model_analyzer_output(self, output: str) -> Dict[str, Any]:
        # Input validations
        assert isinstance(output, str), f"{type(output)=}"
        assert output, "output must be non-empty"

        patterns = {
            "images": re.compile(r"Images:\s+(\d+)"),
            "points": re.compile(r"Points:\s+(\d+)"),
            "observations": re.compile(r"Observations:\s+(\d+)"),
            "mean_reprojection_error_px": re.compile(
                r"Mean reprojection error:\s+([0-9.]+)"
            ),
        }
        metrics: Dict[str, Any] = {}
        for key, pattern in patterns.items():
            match = pattern.search(output)
            assert match is not None, f"model_analyzer missing {key}: {output}"
            if key == "mean_reprojection_error_px":
                metrics[key] = float(match.group(1))
            else:
                metrics[key] = int(match.group(1))
        return metrics
