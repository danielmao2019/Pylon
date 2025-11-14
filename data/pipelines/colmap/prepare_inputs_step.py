"""Step that prepares COLMAP inputs by copying source imagery."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List

from project.pipelines.base_step import BaseStep


class PrepareColmapInputsStep(BaseStep):
    """Copy source JPEGs into the scene's COLMAP input directory."""

    STEP_NAME = "prepare_colmap_inputs"

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        super().__init__(input_root=input_root, output_root=output_root)

    def check_inputs(self) -> None:  # type: ignore[override]
        logging.info("🔍 Validating source directory %s", self.input_root)
        assert (
            self.input_root.is_dir()
        ), f"Source image directory not found: {self.input_root}"
        jpg_names = self._list_image_basenames(self.input_root)
        assert jpg_names, f"No JPG images found in {self.input_root}"
        logging.info("   ✓ Found %d source images", len(jpg_names))

    def check_outputs(self) -> bool:
        if not self.output_root.is_dir():
            return False
        source_names = self._list_image_basenames(self.input_root)
        if not source_names:
            return False
        self.OUTPUT_FILES = list(source_names)
        return super().check_outputs()

    def run(self, force: bool = False) -> None:
        self.check_inputs()
        self.output_root.mkdir(parents=True, exist_ok=True)
        if not force and self.check_outputs():
            logging.info("📥 COLMAP inputs already prepared - SKIPPED")
            return
        self._prepare_inputs()

    def _prepare_inputs(self) -> None:
        logging.info("📸 Copying source JPGs to COLMAP input directory")
        jpg_basenames = self._list_image_basenames(self.input_root)
        assert jpg_basenames, f"No JPG files found in {self.input_root}"
        for filename in jpg_basenames:
            source_path = self.input_root / filename
            input_path = self.output_root / filename
            shutil.copy2(source_path, input_path)
        logging.info("   ✓ Copied %d images", len(jpg_basenames))

    def _list_image_basenames(self, directory: Path) -> List[str]:
        basenames = [
            entry.name
            for entry in directory.iterdir()
            if entry.is_file() and entry.name.lower().endswith(".jpg")
        ]
        return sorted(basenames)
