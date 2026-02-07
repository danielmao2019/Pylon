"""Step that runs COLMAP feature matching."""

import logging
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from data.pipelines.base_step import BaseStep

COLMAP_MAX_IMAGE_ID = 2**31 - 1  # Matches COLMAP Database::kMaxImageId


class ColmapFeatureMatchingStep(BaseStep):
    """Copy from NerfStudio pipeline: COLMAP exhaustive_matcher stage."""

    STEP_NAME = "colmap_feature_matching"

    def __init__(
        self,
        scene_root: str | Path,
        colmap_args: Dict[str, str],
        matcher_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Input validations
        assert matcher_cfg is None or isinstance(
            matcher_cfg, dict
        ), f"{type(matcher_cfg)=}"
        assert (
            matcher_cfg is None or "matcher_type" in matcher_cfg
        ), "matcher_cfg must include matcher_type"
        assert matcher_cfg is None or matcher_cfg["matcher_type"] in {
            "exhaustive_matcher",
            "sequential_matcher",
        }, f"Invalid matcher_type {matcher_cfg['matcher_type']}"
        assert matcher_cfg is None or (
            (
                matcher_cfg["matcher_type"] == "exhaustive_matcher"
                and matcher_cfg.keys() == {"matcher_type"}
            )
            or (
                matcher_cfg["matcher_type"] == "sequential_matcher"
                and matcher_cfg.keys()
                <= {"matcher_type", "overlap", "quadratic_overlap"}
            )
        ), f"Invalid matcher_cfg keys: {matcher_cfg.keys()}"
        assert matcher_cfg is None or (
            matcher_cfg["matcher_type"] != "sequential_matcher"
            or "overlap" not in matcher_cfg
            or matcher_cfg["overlap"] > 0
        ), "matcher_cfg overlap must be positive"

        scene_root = Path(scene_root)
        super().__init__(input_root=scene_root, output_root=scene_root)
        self.input_images_dir = scene_root / "input"
        self.distorted_dir = scene_root / "distorted"
        self.database_path = scene_root / "distorted" / "database.db"
        self.colmap_args = colmap_args
        self.matcher_cfg = matcher_cfg

    def _init_input_files(self) -> None:
        self.input_files = ["distorted/database.db"]

    def _init_output_files(self) -> None:
        self.output_files = ["distorted/database.db"]

    def build(self, force: bool = False) -> None:
        if self._built:
            return
        super().build(force=force)
        self.run(kwargs={}, force=force)

    def check_outputs(self) -> bool:
        """Return True only when the COLMAP database contains match entries."""
        if not super().check_outputs():
            return False
        try:
            self._validate_database()
        except Exception as e:
            logging.debug("Feature matching validation failed: %s", e)
            return False
        else:
            return True

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        if self.check_outputs() and not force:
            return {}

        self.distorted_dir.mkdir(parents=True, exist_ok=True)

        logging.info("   🔗 Feature matching")
        cmd_parts = self._build_colmap_command()
        result = subprocess.run(cmd_parts, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"COLMAP feature matching failed with code {result.returncode}. "
            f"Using {self.colmap_args['version']} with parameters: "
            f"{self.colmap_args['matching_use_gpu']}, {self.colmap_args['guided_matching']}. "
            f"STDOUT: {result.stdout} STDERR: {result.stderr}"
        )

        self._validate_database()
        return {}

    def _build_colmap_command(self) -> List[str]:
        if (
            self.matcher_cfg is None
            or self.matcher_cfg["matcher_type"] == "exhaustive_matcher"
        ):
            cmd_parts = [
                "colmap",
                "exhaustive_matcher",
                "--database_path",
                str(self.database_path),
                self.colmap_args["matching_use_gpu"],
                "1",
                self.colmap_args["guided_matching"],
                "1",
                "--log_to_stderr",
                "1",
            ]
        else:
            cmd_parts = [
                "colmap",
                "sequential_matcher",
                "--database_path",
                str(self.database_path),
                self.colmap_args["matching_use_gpu"],
                "1",
                self.colmap_args["guided_matching"],
                "1",
                "--log_to_stderr",
                "1",
            ]
            for key in self.matcher_cfg:
                if key == "matcher_type":
                    continue
                cmd_parts.extend(
                    [f"--SequentialMatching.{key}", str(self.matcher_cfg[key])]
                )
        return cmd_parts

    def _validate_database(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.cursor()
            image_ids = self._validate_images_table(cursor)
            self._validate_matches_table(cursor)
            geometry_pair_ids = self._validate_two_view_geometries_table(cursor)
            self._assert_pair_ids_reference_images(geometry_pair_ids, image_ids)

    def _validate_images_table(self, cursor: sqlite3.Cursor) -> Set[int]:
        image_rows = cursor.execute("SELECT image_id, name FROM images").fetchall()
        assert image_rows, f"No images recorded in database {self.database_path}"
        image_names = [row[1] for row in image_rows]

        input_paths = sorted(self.input_images_dir.iterdir())
        assert input_paths, f"Empty input dir or no files: {self.input_images_dir}"
        assert all(entry.is_file() for entry in input_paths), (
            "COLMAP input directory must only contain files "
            f"(found non-file entries in {self.input_images_dir})"
        )
        assert all(entry.suffix == ".png" for entry in input_paths), (
            f"Non-PNG files present in COLMAP input directory: "
            f"{', '.join(sorted(entry.name for entry in input_paths if entry.suffix != '.png'))}"
        )
        expected_names = [entry.name for entry in input_paths]

        assert sorted(image_names) == expected_names, (
            "Image names in database do not match COLMAP inputs. "
            f"expected={len(expected_names)} actual={len(image_names)}"
        )
        return {row[0] for row in image_rows}

    def _validate_matches_table(self, cursor: sqlite3.Cursor) -> None:
        match_rows = cursor.execute(
            "SELECT pair_id, rows, data FROM matches"
        ).fetchall()
        assert match_rows, f"No matches stored in database {self.database_path}"
        invalid_rows = [
            row
            for row in match_rows
            if row[1] == 0 or row[2] is None or len(row[2]) == 0
        ]
        valid_match_ratio = (len(match_rows) - len(invalid_rows)) / len(match_rows)
        assert valid_match_ratio > 0.10, (
            "matches table contains too many invalid entries. "
            f"total_rows={len(match_rows)} invalid_pairs={len(invalid_rows)} "
            f"valid_ratio={valid_match_ratio:.4f} threshold=0.10 "
            f"sample_pair_ids={sorted(row[0] for row in invalid_rows)[:3]}"
        )

    def _validate_two_view_geometries_table(self, cursor: sqlite3.Cursor) -> List[int]:
        geometry_rows = cursor.execute(
            "SELECT pair_id, rows, data, config FROM two_view_geometries"
        ).fetchall()
        assert (
            geometry_rows
        ), f"No two_view_geometries rows in database {self.database_path}"
        invalid_rows = [
            row
            for row in geometry_rows
            if row[1] == 0 or row[2] is None or len(row[2]) == 0 or row[3] == 0
        ]
        valid_geometry_ratio = (len(geometry_rows) - len(invalid_rows)) / len(
            geometry_rows
        )
        assert valid_geometry_ratio > 0.10, (
            "two_view_geometries contains too many invalid entries. "
            f"total_rows={len(geometry_rows)} invalid_pairs={len(invalid_rows)} "
            f"valid_ratio={valid_geometry_ratio:.4f} threshold=0.10 "
            f"sample_pair_ids={sorted(row[0] for row in invalid_rows)[:3]}"
        )
        pair_ids = [row[0] for row in geometry_rows]
        assert all(pid > 0 for pid in pair_ids), "pair_id values must be positive"
        return pair_ids

    def _assert_pair_ids_reference_images(
        self, pair_ids: List[int], image_ids: Set[int]
    ) -> None:
        for pair_id in pair_ids:
            image_id2 = pair_id % COLMAP_MAX_IMAGE_ID
            image_id1 = (pair_id - image_id2) // COLMAP_MAX_IMAGE_ID
            assert image_id1 < image_id2, (
                f"pair_id {pair_id} decoded to unordered image ids "
                f"({image_id1}, {image_id2})"
            )
            assert image_id1 in image_ids and image_id2 in image_ids, (
                f"pair_id {pair_id} references images "
                f"({image_id1}, {image_id2}) not present in images table"
            )
