"""Step that runs COLMAP bundle_adjuster on the initialized model."""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

from data.pipelines.base_step import BaseStep
from data.structures.colmap.load import (
    _load_colmap_cameras_bin,
    _load_colmap_images_bin,
)


class ColmapBundleAdjustmentStep(BaseStep):
    """Refine the initialized sparse model using bundle adjustment."""

    STEP_NAME = "colmap_bundle_adjustment"

    def __init__(self, scene_root: str | Path) -> None:
        scene_root = Path(scene_root)
        self.model_dir = scene_root / "distorted" / "sparse" / "0"
        super().__init__(input_root=scene_root, output_root=scene_root)

    def _init_input_files(self) -> None:
        self.input_files = [
            "distorted/sparse/0/cameras.bin",
            "distorted/sparse/0/images.bin",
            "distorted/sparse/0/points3D.bin",
        ]

    def _init_output_files(self) -> None:
        self.output_files = [
            "distorted/sparse/0/cameras.bin",
            "distorted/sparse/0/images.bin",
            "distorted/sparse/0/points3D.bin",
        ]

    def check_outputs(self) -> bool:
        outputs_ready = super().check_outputs()
        if not outputs_ready:
            return False
        try:
            self._validate_model()
            return True
        except Exception as e:
            logging.debug("Bundle adjustment validation failed: %s", e)
            return False

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        if not force and self.check_outputs():
            logging.info("   📏 COLMAP bundle adjustment already done - SKIPPED")
            return {}
        cmd_parts = [
            "colmap",
            "bundle_adjuster",
            "--input_path",
            str(self.model_dir),
            "--output_path",
            str(self.model_dir),
            "--BundleAdjustment.refine_principal_point",
            "0",
            "--BundleAdjustment.refine_focal_length",
            "1",
            "--BundleAdjustment.refine_extra_params",
            "1",
            "--log_to_stderr",
            "1",
        ]
        result = subprocess.run(cmd_parts, capture_output=True, text=True)
        ret_code = result.returncode
        assert ret_code == 0, (
            f"COLMAP bundle_adjuster failed with code {ret_code} "
            f"for model {self.model_dir}"
        )
        self._validate_model()
        return {}

    def _validate_model(self) -> None:
        cameras = _load_colmap_cameras_bin(
            path_to_model_file=str(self.model_dir / "cameras.bin")
        )
        images = _load_colmap_images_bin(
            path_to_model_file=str(self.model_dir / "images.bin")
        )
        assert cameras, f"No cameras parsed from {self.model_dir / 'cameras.bin'}"
        assert (
            len(cameras) == 1
        ), f"Expected exactly one camera, got {len(cameras)} in {self.model_dir}"
        assert (
            images
        ), f"No registered images parsed from {self.model_dir / 'images.bin'}"
