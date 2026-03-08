"""Step that runs COLMAP feature extraction."""

import logging
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.pipelines.base_step import BaseStep


class ColmapFeatureExtractionStep(BaseStep):
    """Copy from NerfStudio pipeline: COLMAP feature_extractor stage."""

    STEP_NAME = "colmap_feature_extraction"

    def __init__(
        self,
        scene_root: str | Path,
        colmap_args: Dict[str, str],
        extractor_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Input validations
        assert isinstance(scene_root, (str, Path)), (
            "scene_root must be str or Path. " f"{type(scene_root)=} {scene_root=}"
        )
        assert isinstance(colmap_args, dict), (
            "colmap_args must be a dict. " f"{type(colmap_args)=}"
        )
        assert extractor_cfg is None or isinstance(extractor_cfg, dict), (
            "extractor_cfg must be None or dict. " f"{type(extractor_cfg)=}"
        )
        assert extractor_cfg is None or extractor_cfg.keys() <= {
            "upright",
            "camera_mode",
            "mask_input_root",
        }, (
            "extractor_cfg contains unsupported keys. "
            f"keys={sorted(extractor_cfg.keys())}"
        )
        assert (
            extractor_cfg is None
            or "mask_input_root" not in extractor_cfg
            or extractor_cfg["mask_input_root"] is None
            or isinstance(extractor_cfg["mask_input_root"], (str, Path))
        ), (
            "extractor_cfg['mask_input_root'] must be None, str, or Path. "
            f"value={extractor_cfg['mask_input_root']} "
            f"type={type(extractor_cfg['mask_input_root'])}"
        )
        assert (
            extractor_cfg is None
            or "upright" not in extractor_cfg
            or isinstance(extractor_cfg["upright"], bool)
        ), (
            "extractor_cfg['upright'] must be a bool when provided. "
            f"value={extractor_cfg['upright']} type={type(extractor_cfg['upright'])}"
        )
        assert (
            extractor_cfg is None
            or "camera_mode" not in extractor_cfg
            or (
                isinstance(extractor_cfg["camera_mode"], str)
                and extractor_cfg["camera_mode"]
                in {"SIMPLE_PINHOLE", "PINHOLE", "OPENCV"}
            )
        ), (
            "extractor_cfg['camera_mode'] must be a str in "
            "{'SIMPLE_PINHOLE', 'PINHOLE', 'OPENCV'} when provided. "
            f"value={extractor_cfg['camera_mode']} type={type(extractor_cfg['camera_mode'])}"
        )
        assert (
            extractor_cfg is None
            or "mask_input_root" not in extractor_cfg
            or "mask_path" in colmap_args
        ), (
            "colmap_args must include 'mask_path' when mask_input_root is configured. "
            f"colmap_arg_keys={sorted(colmap_args.keys())}"
        )

        # Input normalizations
        scene_root = Path(scene_root)

        mask_input_root = None
        if extractor_cfg is not None and "mask_input_root" in extractor_cfg:
            if extractor_cfg["mask_input_root"] is not None:
                mask_input_root = Path(extractor_cfg["mask_input_root"])

        self.input_dir = scene_root / "input"
        self.masks_dir = mask_input_root
        self.distorted_dir = scene_root / "distorted"
        self.database_path = scene_root / "distorted" / "database.db"
        self.colmap_args = colmap_args
        self.extractor_cfg = extractor_cfg
        super().__init__(input_root=scene_root, output_root=scene_root)

    def _init_input_files(self) -> None:
        entries = sorted(self.input_dir.iterdir())
        self.input_files = [f"input/{entry.name}" for entry in entries]

    def _init_output_files(self) -> None:
        self.output_files = ["distorted/database.db"]

    def build(self, force: bool = False) -> None:
        if self._built:
            return
        super().build(force=force)
        self.run(kwargs={}, force=force)

    def check_inputs(self) -> None:
        super().check_inputs()
        _ = self._build_expected_image_names()

    def check_outputs(self) -> bool:
        outputs_ready = super().check_outputs()
        if not outputs_ready:
            return False
        try:
            self._validate_database()
        except Exception as e:
            logging.debug("Feature extraction validation failed: %s", e)
            return False
        else:
            return True

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        if not force and self.check_outputs():
            return {}

        self.distorted_dir.mkdir(parents=True, exist_ok=True)
        if self.database_path.exists():
            self.database_path.unlink()

        logging.info("   🔍 Feature extraction")
        cmd_parts = self._build_colmap_command()
        result = subprocess.run(cmd_parts, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"COLMAP feature extraction failed with code {result.returncode}. "
            f"Using {self.colmap_args['version']} with parameter: {self.colmap_args['feature_use_gpu']}. "
            f"STDOUT: {result.stdout} STDERR: {result.stderr}"
        )

        self._validate_database()
        return {}

    def _build_colmap_command(self) -> List[str]:
        cmd_parts = [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(self.database_path),
            "--image_path",
            str(self.input_dir),
            "--ImageReader.single_camera",
            "1",
            self.colmap_args["feature_use_gpu"],
            "1",
            "--log_to_stderr",
            "1",
        ]
        if self.extractor_cfg is not None and "camera_mode" in self.extractor_cfg:
            cmd_parts.extend(
                ["--ImageReader.camera_model", self.extractor_cfg["camera_mode"]]
            )
        if (
            self.extractor_cfg is not None
            and "mask_input_root" in self.extractor_cfg
            and self.extractor_cfg["mask_input_root"] is not None
        ):
            cmd_parts.extend(
                [
                    self.colmap_args["mask_path"],
                    str(self.extractor_cfg["mask_input_root"]),
                ]
            )
        if self.extractor_cfg is not None and "upright" in self.extractor_cfg:
            cmd_parts.extend(
                [
                    self.colmap_args["upright"],
                    "1" if self.extractor_cfg["upright"] else "0",
                ]
            )
        return cmd_parts

    def _validate_database(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.cursor()
            image_ids = self._validate_images_table(cursor=cursor)
            self._validate_keypoints_table(cursor=cursor, image_ids=image_ids)
            self._validate_descriptors_table(cursor=cursor, image_ids=image_ids)
            self._validate_cameras_table(cursor=cursor)

    def _validate_images_table(self, cursor: sqlite3.Cursor) -> set[int]:
        # Input validations
        assert isinstance(cursor, sqlite3.Cursor), f"{type(cursor)=}"
        assert isinstance(self.input_files, list), f"{type(self.input_files)=}"

        image_rows = cursor.execute("SELECT image_id, name FROM images").fetchall()
        assert image_rows, f"No images recorded in database {self.database_path}"
        image_names = [row[1] for row in image_rows]

        expected_names = self._build_expected_image_names()

        assert sorted(image_names) == expected_names, (
            "Image names in database do not match COLMAP inputs. "
            f"expected={len(expected_names)} actual={len(image_names)}"
        )
        return {row[0] for row in image_rows}

    def _validate_keypoints_table(
        self, cursor: sqlite3.Cursor, image_ids: set[int]
    ) -> None:
        # Input validations
        assert isinstance(cursor, sqlite3.Cursor), f"{type(cursor)=}"
        assert isinstance(image_ids, set), f"{type(image_ids)=}"
        assert image_ids, "image_ids must be non-empty"

        keypoint_rows = cursor.execute(
            "SELECT image_id, rows FROM keypoints"
        ).fetchall()
        assert keypoint_rows, f"No keypoints stored in database {self.database_path}"
        assert all(
            row[1] > 0 for row in keypoint_rows
        ), "Keypoints row count must be positive for all images"
        keypoint_ids = {row[0] for row in keypoint_rows}
        assert (
            keypoint_ids == image_ids
        ), "Keypoints table image_ids do not match images table"

    def _validate_descriptors_table(
        self, cursor: sqlite3.Cursor, image_ids: set[int]
    ) -> None:
        # Input validations
        assert isinstance(cursor, sqlite3.Cursor), f"{type(cursor)=}"
        assert isinstance(image_ids, set), f"{type(image_ids)=}"
        assert image_ids, "image_ids must be non-empty"

        descriptor_rows = cursor.execute("SELECT image_id FROM descriptors").fetchall()
        assert (
            descriptor_rows
        ), f"No descriptors stored in database {self.database_path}"
        descriptor_ids = {row[0] for row in descriptor_rows}
        assert (
            descriptor_ids == image_ids
        ), "Descriptor image_ids do not match images table"

    def _validate_cameras_table(self, cursor: sqlite3.Cursor) -> None:
        # Input validations
        assert isinstance(cursor, sqlite3.Cursor), f"{type(cursor)=}"

        camera_count = cursor.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
        assert (
            camera_count == 1
        ), f"Expected exactly one camera entry, found {camera_count}"

    def _build_expected_image_names(self) -> List[str]:
        input_paths = sorted(self.input_dir.iterdir())
        assert input_paths, f"Empty input dir or no files: {self.input_dir}"
        assert all(path.is_file() for path in input_paths), (
            "COLMAP input directory must only contain files "
            f"(found non-file entries in {self.input_dir})"
        )
        assert all(path.suffix == ".png" for path in input_paths), (
            f"Non-PNG files present in COLMAP input directory: "
            f"{', '.join(sorted(path.name for path in input_paths if path.suffix != '.png'))}"
        )
        input_names = sorted(path.name for path in input_paths)

        if self.masks_dir is None:
            return input_names

        assert self.masks_dir.is_dir(), f"{self.masks_dir=}"
        mask_paths = sorted(self.masks_dir.iterdir())
        assert mask_paths, f"{self.masks_dir=}"
        assert all(path.is_file() for path in mask_paths), (
            "COLMAP mask directory must only contain files "
            f"(found non-file entries in {self.masks_dir})"
        )
        assert all(path.suffix == ".png" for path in mask_paths), (
            f"Non-PNG files present in COLMAP mask directory: "
            f"{', '.join(sorted(path.name for path in mask_paths if path.suffix != '.png'))}"
        )
        mask_names = {path.name for path in mask_paths}

        expected_names = sorted(name for name in input_names if name in mask_names)
        assert expected_names, (
            "No usable COLMAP frames after input-mask intersection. "
            f"num_inputs={len(input_names)} num_masks={len(mask_names)}"
        )
        return expected_names
