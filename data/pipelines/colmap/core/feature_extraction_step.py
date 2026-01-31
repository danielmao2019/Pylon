"""Step that runs COLMAP feature extraction."""

import logging
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Dict

from data.pipelines.base_step import BaseStep


class ColmapFeatureExtractionStep(BaseStep):
    """Copy from NerfStudio pipeline: COLMAP feature_extractor stage."""

    STEP_NAME = "colmap_feature_extraction"

    def __init__(
        self,
        scene_root: str | Path,
        colmap_args: Dict[str, str],
        upright: bool = False,
        camera_mode: str = "OPENCV",
        mask_input_root: str | Path | None = None,
    ) -> None:
        # Input validations
        assert isinstance(scene_root, (str, Path)), f"{type(scene_root)=}"
        assert isinstance(colmap_args, dict), f"{type(colmap_args)=}"
        assert isinstance(upright, bool), f"{type(upright)=}"
        assert isinstance(camera_mode, str), f"{type(camera_mode)=}"
        assert camera_mode in {
            "SIMPLE_PINHOLE",
            "PINHOLE",
            "OPENCV",
        }, f"{camera_mode=}"
        assert mask_input_root is None or isinstance(
            mask_input_root, (str, Path)
        ), f"{type(mask_input_root)=}"

        scene_root = Path(scene_root)
        self.input_images_dir = scene_root / "input"
        self.distorted_dir = scene_root / "distorted"
        self.database_path = scene_root / "distorted" / "database.db"
        self.colmap_args = colmap_args
        self.upright = upright
        self.camera_mode = camera_mode
        self.mask_input_root = mask_input_root
        super().__init__(input_root=scene_root, output_root=scene_root)

    def _init_input_files(self) -> None:
        entries = sorted(self.input_images_dir.iterdir())
        self.input_files = [f"input/{entry.name}" for entry in entries]

    def _init_output_files(self) -> None:
        self.output_files = ["distorted/database.db"]

    def build(self, force: bool = False) -> None:
        super().build(force=force)
        self.run(kwargs={}, force=force)

    def check_inputs(self) -> None:
        super().check_inputs()
        entries = sorted(self.input_images_dir.iterdir())
        assert all(entry.is_file() for entry in entries), (
            "COLMAP input directory must only contain files "
            f"(found non-file entries in {self.input_images_dir})"
        )
        assert all(entry.suffix == ".png" for entry in entries), (
            f"Non-PNG files present in COLMAP input directory: "
            f"{', '.join(sorted(entry.name for entry in entries if entry.suffix != '.png'))}"
        )

    def check_outputs(self) -> bool:
        outputs_ready = super().check_outputs()
        if not outputs_ready:
            return False
        try:
            self._validate_database_contents()
        except Exception as e:
            logging.debug("Feature extraction validation failed: %s", e)
            return False
        else:
            return True

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.distorted_dir.mkdir(parents=True, exist_ok=True)
        if not force and self.check_outputs():
            logging.info("📥 COLMAP feature extraction already done - SKIPPED")
            return {}
        if self.database_path.exists():
            self.database_path.unlink()
        logging.info("   🔍 Feature extraction")
        cmd_parts = [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(self.database_path),
            "--image_path",
            str(self.input_images_dir),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            self.camera_mode,
            self.colmap_args["feature_use_gpu"],
            "1",
            "--log_to_stderr",
            "1",
        ]
        if self.mask_input_root is not None:
            cmd_parts.extend([self.colmap_args["mask_path"], str(self.mask_input_root)])
        if self.upright:
            cmd_parts.extend([self.colmap_args["upright"], "1"])
        result = subprocess.run(cmd_parts, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"COLMAP feature extraction failed with code {result.returncode}. "
            f"Using {self.colmap_args['version']} with parameter: {self.colmap_args['feature_use_gpu']}. "
            f"STDOUT: {result.stdout} STDERR: {result.stderr}"
        )
        self._validate_database_contents()
        return {}

    def _validate_database_contents(self) -> None:
        input_paths = [self.input_root / rel for rel in self.input_files]
        assert input_paths, f"Empty input dir or no files: {self.input_images_dir}"
        assert all(path.is_file() for path in input_paths), (
            "COLMAP input directory must only contain files "
            f"(found non-file entries in {self.input_images_dir})"
        )
        assert all(path.suffix == ".png" for path in input_paths), (
            f"Non-PNG files present in COLMAP input directory: "
            f"{', '.join(sorted(path.name for path in input_paths if path.suffix != '.png'))}"
        )
        expected_names = sorted(path.name for path in input_paths)

        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.cursor()

            image_rows = cursor.execute("SELECT image_id, name FROM images").fetchall()
            assert image_rows, f"No images recorded in database {self.database_path}"
            image_names = [row[1] for row in image_rows]
            assert sorted(image_names) == expected_names, (
                "Image names in database do not match COLMAP inputs. "
                f"expected={len(expected_names)} actual={len(image_names)}"
            )
            image_ids = {row[0] for row in image_rows}

            keypoint_rows = cursor.execute(
                "SELECT image_id, rows FROM keypoints"
            ).fetchall()
            assert (
                keypoint_rows
            ), f"No keypoints stored in database {self.database_path}"
            assert all(
                row[1] > 0 for row in keypoint_rows
            ), "Keypoints row count must be positive for all images"
            keypoint_ids = {row[0] for row in keypoint_rows}
            assert (
                keypoint_ids == image_ids
            ), "Keypoints table image_ids do not match images table"

            descriptor_rows = cursor.execute(
                "SELECT image_id FROM descriptors"
            ).fetchall()
            assert (
                descriptor_rows
            ), f"No descriptors stored in database {self.database_path}"
            descriptor_ids = {row[0] for row in descriptor_rows}
            assert (
                descriptor_ids == image_ids
            ), "Descriptor image_ids do not match images table"

            camera_count = cursor.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
            assert (
                camera_count == 1
            ), f"Expected exactly one camera entry, found {camera_count}"
