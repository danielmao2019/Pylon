"""Step that writes COLMAP txt models from binary outputs."""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from data.pipelines.base_step import BaseStep


class ColmapModelTxtExportStep(BaseStep):
    """Export COLMAP cameras/images/points as txt files from binary models."""

    STEP_NAME = "colmap_model_txt_export"

    def __init__(self, scene_root: str | Path, model_relpath: str | Path) -> None:
        scene_root = Path(scene_root)
        self.model_relpath = Path(model_relpath)
        self.model_dir = scene_root / self.model_relpath
        super().__init__(input_root=scene_root, output_root=scene_root)

    def _init_input_files(self) -> None:
        rel_root = self.model_relpath.as_posix()
        self.input_files = [
            f"{rel_root}/cameras.bin",
            f"{rel_root}/images.bin",
            f"{rel_root}/points3D.bin",
        ]

    def _init_output_files(self) -> None:
        rel_root = self.model_relpath.as_posix()
        self.output_files = [
            f"{rel_root}/cameras.txt",
            f"{rel_root}/images.txt",
            f"{rel_root}/points3D.txt",
        ]

    def check_outputs(self) -> bool:
        outputs_ready = super().check_outputs()
        if not outputs_ready:
            return False
        try:
            self._validate_sparse_model_dir()
        except Exception as e:
            logging.debug("COLMAP txt export validation failed: %s", e)
            return False
        return True

    def _validate_sparse_model_dir(self) -> None:
        cameras_path = self.model_dir / "cameras.bin"
        images_path = self.model_dir / "images.bin"
        points_path = self.model_dir / "points3D.bin"
        assert cameras_path.exists(), f"cameras.bin not found: {cameras_path}"
        assert images_path.exists(), f"images.bin not found: {images_path}"
        assert points_path.exists(), f"points3D.bin not found: {points_path}"

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        if not force and self.check_outputs():
            return {}

        self._validate_sparse_model_dir()
        logging.info("   📄 Exporting COLMAP txt model: %s", self.model_dir)
        cmd_parts = self._build_colmap_command()
        result = subprocess.run(cmd_parts, capture_output=True, text=True)
        assert result.returncode == 0, (
            "COLMAP model_converter failed with code "
            f"{result.returncode} for model {self.model_dir}. "
            f"STDOUT: {result.stdout} STDERR: {result.stderr}"
        )
        return {}

    def _build_colmap_command(self) -> List[str]:
        return [
            "colmap",
            "model_converter",
            "--input_path",
            str(self.model_dir),
            "--output_path",
            str(self.model_dir),
            "--output_type",
            "TXT",
            "--log_to_stderr",
            "1",
        ]
