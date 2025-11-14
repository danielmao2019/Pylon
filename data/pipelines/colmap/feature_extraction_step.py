"""Step that runs COLMAP feature extraction."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict

from project.pipelines.base_step import BaseStep


class ColmapFeatureExtractionStep(BaseStep):
    """Copy from NerfStudio pipeline: COLMAP feature_extractor stage."""

    STEP_NAME = "colmap_feature_extraction"
    OUTPUT_FILES = ["database.db"]

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        super().__init__(input_root=input_root, output_root=output_root)
        self.database_path = self.output_root / "database.db"

    def run(self, force: bool = False) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        if not force and self.check_outputs():
            logging.info("📥 COLMAP feature extraction already done - SKIPPED")
            return
        colmap_params = self._get_colmap_parameters()
        logging.info("   🔍 Feature extraction")
        feat_extraction_cmd = (
            f"colmap feature_extractor "
            f"--database_path {self.database_path} "
            f"--image_path {self.input_root} "
            f"--ImageReader.single_camera 1 "
            f"--ImageReader.camera_model OPENCV "
            f"{colmap_params['feature_use_gpu']} 1"
        )
        ret_code = subprocess.call(feat_extraction_cmd, shell=True)
        assert ret_code == 0, (
            f"COLMAP feature extraction failed with code {ret_code}. "
            f"Using {colmap_params['version']} with parameter: {colmap_params['feature_use_gpu']}"
        )

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
