"""Step to convert COLMAP outputs into NerfStudio data."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from data.pipelines.base_step import BaseStep
from data.structures.three_d.colmap.colmap_data import COLMAP_Data
from data.structures.three_d.colmap.convert import convert_colmap_to_nerfstudio
from data.structures.three_d.nerfstudio.nerfstudio_data import NerfStudio_Data
from data.structures.three_d.nerfstudio.validate import MODALITY_SPECS


class ColmapConvertToNerfstudioStep(BaseStep):
    """Export NerfStudio JSON and sparse point cloud from COLMAP outputs."""

    STEP_NAME = "colmap_convert_to_nerfstudio"

    def __init__(self, input_root: str | Path, output_root: str | Path) -> None:
        super().__init__(input_root=input_root, output_root=output_root)
        self.transforms_path = self.output_root / "transforms.json"

    def _init_input_files(self) -> None:
        self.input_files = ["0/cameras.bin", "0/images.bin", "0/points3D.bin"]

    def _init_output_files(self) -> None:
        self.output_files = ["transforms.json", "sparse_pc.ply"]

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
        if not force and self.check_outputs():
            return {}

        self.output_root.mkdir(parents=True, exist_ok=True)
        model_dir = self.input_root / "0"
        colmap_data = COLMAP_Data.load(model_dir=model_dir)
        transforms_path, ply_path = convert_colmap_to_nerfstudio(
            filename="transforms.json",
            colmap_cameras=colmap_data.cameras,
            colmap_images=colmap_data.images,
            colmap_points=colmap_data.points3D,
            output_dir=str(self.output_root),
            ply_filename="sparse_pc.ply",
        )
        logging.info(
            "   ✓ Wrote %s with %d frames", transforms_path, len(colmap_data.images)
        )
        logging.info("   ✓ Wrote COLMAP sparse point cloud: %s", ply_path)
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
