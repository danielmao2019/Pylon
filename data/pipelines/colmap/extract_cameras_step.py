"""Step to extract COLMAP camera intrinsics and extrinsics."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from data.pipelines.base_step import BaseStep
from data.structures.three_d.colmap.colmap_data import COLMAP_Data
from data.structures.three_d.colmap.convert import create_nerfstudio_from_colmap
from data.structures.three_d.nerfstudio.nerfstudio_data import NerfStudio_Data
from data.structures.three_d.nerfstudio.validate import MODALITY_SPECS


class ColmapExtractCamerasStep(BaseStep):
    """Export complete NerfStudio_Data JSON from COLMAP outputs."""

    STEP_NAME = "colmap_extract_cameras"

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        super().__init__(input_root=input_root, output_root=output_root)
        self.model_dir = self.input_root / "0"
        self.transforms_path = self.output_root / "transforms.json"

    def _init_input_files(self) -> None:
        self.input_files = ["0/cameras.bin", "0/images.bin"]

    def _init_output_files(self) -> None:
        self.output_files = ["transforms.json"]

    def check_outputs(self) -> bool:
        outputs_ready = super().check_outputs()
        if not outputs_ready:
            return False
        try:
            nerfstudio = NerfStudio_Data.load(
                filepath=self.transforms_path, device=torch.device("cpu")
            )
            self._validate_conversion(
                modalities=nerfstudio.modalities,
                filenames=nerfstudio.filenames,
            )
        except Exception as e:
            logging.debug("COLMAP NerfStudio_Data validation failed: %s", e)
            return False
        return True

    def run(self, kwargs: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        self.check_inputs()
        self.output_root.mkdir(parents=True, exist_ok=True)
        if not force and self.check_outputs():
            logging.info("🎥 COLMAP cameras already extracted - SKIPPED")
            return {}

        colmap_data = COLMAP_Data.load(model_dir=self.model_dir)
        create_nerfstudio_from_colmap(
            filename="transforms.json",
            colmap_cameras=colmap_data.cameras,
            colmap_images=colmap_data.images,
            output_dir=str(self.output_root),
            ply_file_path="sparse_pc.ply",
        )
        logging.info(
            "   ✓ Wrote transforms.json with %d frames", len(colmap_data.images)
        )
        return {}

    def _validate_conversion(self, modalities: List[str], filenames: List[str]) -> None:
        # Input validations
        assert isinstance(modalities, list), f"{type(modalities)=}"
        assert modalities, "modalities must be non-empty"
        assert all(isinstance(item, str) for item in modalities), f"{modalities=}"
        assert isinstance(filenames, list), f"{type(filenames)=}"
        assert filenames, "filenames must be non-empty"
        assert all(isinstance(item, str) for item in filenames), f"{filenames=}"

        disk_modalities, disk_filenames = self._get_disk_modalities_filenames()
        assert set(disk_modalities) == set(modalities), (
            "Modalities on disk do not match transforms.json: "
            f"disk={disk_modalities} expected={modalities}"
        )
        assert set(disk_filenames) == set(filenames), (
            "Filenames on disk do not match transforms.json: "
            f"disk={len(disk_filenames)} expected={len(filenames)}"
        )

    def _get_disk_modalities_filenames(self) -> Tuple[List[str], List[str]]:
        # Input validations
        assert self.output_root.is_dir(), f"{self.output_root=}"

        disk_modalities: List[str] = []
        disk_filenames: List[str] = []
        filenames_by_modality: Dict[str, List[str]] = {}

        for modality, spec in MODALITY_SPECS.items():
            modality_folder = spec[1]
            modality_dir = self.output_root / modality_folder
            if not modality_dir.is_dir():
                continue
            disk_modalities.append(modality)
            modality_files = sorted(
                entry.name for entry in modality_dir.glob("*.png") if entry.is_file()
            )
            assert modality_files, f"No images found in {modality_dir}"
            unique_files = set(modality_files)
            assert len(unique_files) == len(modality_files), (
                "Duplicate image filenames present on disk: "
                f"{', '.join(sorted(name for name in modality_files if modality_files.count(name) > 1))}"
            )
            filenames_by_modality[modality] = [
                Path(name).stem for name in modality_files
            ]

        assert disk_modalities, "No modality folders found on disk"
        assert "image" in disk_modalities, "Image modality folder must exist"
        image_filenames = set(filenames_by_modality["image"])
        assert image_filenames, "No filenames found for image modality"
        for modality in disk_modalities:
            assert set(filenames_by_modality[modality]) == image_filenames, (
                "Modality subfolders must contain identical filenames: "
                f"image vs {modality}"
            )

        disk_filenames = sorted(image_filenames)
        return disk_modalities, disk_filenames
