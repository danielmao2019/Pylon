"""Step that writes COLMAP txt models from binary outputs."""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

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

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        if self.check_outputs() and not force:
            return {}

        logging.info("   📄 Exporting COLMAP txt model: %s", self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        cmd_parts = [
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
        result = subprocess.run(cmd_parts, capture_output=True, text=True)
        assert result.returncode == 0, (
            "COLMAP model_converter failed with code "
            f"{result.returncode} for model {self.model_dir}. "
            f"STDOUT: {result.stdout} STDERR: {result.stderr}"
        )
        return {}
