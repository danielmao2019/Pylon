"""Step that runs COLMAP feature matching."""

from __future__ import annotations

import logging
import subprocess
import sqlite3
from pathlib import Path
from typing import Dict

from project.pipelines.base_step import BaseStep


class ColmapFeatureMatchingStep(BaseStep):
    """Copy from NerfStudio pipeline: COLMAP exhaustive_matcher stage."""

    STEP_NAME = "colmap_feature_matching"
    OUTPUT_FILES = ["database.db"]

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        super().__init__(input_root=input_root, output_root=output_root)
        self.database_path = self.output_root / "database.db"

    def run(self, force: bool = False) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        if not force and self.check_outputs():
            logging.info("🔗 COLMAP feature matching already done - SKIPPED")
            return
        colmap_params = self._get_colmap_parameters()
        logging.info("   🔗 Feature matching")
        feat_matching_cmd = (
            f"colmap exhaustive_matcher "
            f"--database_path {self.database_path} "
            f"{colmap_params['matching_use_gpu']} 1 "
            f"{colmap_params['guided_matching']} 1"
        )
        ret_code = subprocess.call(feat_matching_cmd, shell=True)
        assert ret_code == 0, (
            f"COLMAP feature matching failed with code {ret_code}. "
            f"Using {colmap_params['version']} with parameters: "
            f"{colmap_params['matching_use_gpu']}, {colmap_params['guided_matching']}"
        )

    def check_outputs(self) -> bool:
        """Return True only when the COLMAP database contains match entries."""
        if not self.database_path.exists():
            return False
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.cursor()
            matches_count = cursor.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            geometries_count = cursor.execute(
                "SELECT COUNT(*) FROM two_view_geometries"
            ).fetchone()[0]
        return matches_count > 0 and geometries_count > 0

    def _get_colmap_parameters(self) -> Dict[str, str]:
        try:
            base_help = subprocess.run(
                ["colmap", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("COLMAP executable not found in PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Timed out while querying COLMAP help output") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("Failed to query COLMAP help output") from exc

        version_match = None
        for line in base_help.stdout.splitlines():
            if line.startswith("COLMAP"):
                version_match = line.split()[1]
                break
        if version_match is None:
            raise RuntimeError("Unable to parse COLMAP version from help output")

        version = version_match
        if version == "3.13.0.dev0":
            return {
                "version": version,
                "feature_use_gpu": "--FeatureExtraction.use_gpu",
                "matching_use_gpu": "--FeatureMatching.use_gpu",
                "guided_matching": "--FeatureMatching.guided_matching",
            }
        return {
            "version": version,
            "feature_use_gpu": "--SiftExtraction.use_gpu",
            "matching_use_gpu": "--SiftMatching.use_gpu",
            "guided_matching": "--SiftMatching.guided_matching",
        }
